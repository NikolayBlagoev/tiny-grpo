from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
    GenerationConfig,
)
from sys import argv
import torch.distributed as dist
import torch
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from generate_rollouts import generate_mixed, generate_benign
from utils import trim_, Experience
from reward import reward_answer_binary
from eval_success import eval_asr
from trainer import post_train
from datasets import load_dataset
from attacks import hail_thief
seed = 42
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29501"
device_index = int(argv[1])
malicious = argv[2] == "1"
func = generate_benign
if malicious:
    func = generate_mixed
kl = len(argv) > 3
world_size = 2
dist.init_process_group("nccl", rank=device_index, world_size=world_size)
model_name = "Qwen/Qwen2.5-1.5B"

train_batch_size = 4
lr = 5e-6
kl_weight = 0.01

clean_data = 12
poisoned_data = 9
group_size = 12
my_size = clean_data
if malicious:
    my_size = poisoned_data

poisoned_rollouts = 4
rollouts_per_step = 8


device = f"cuda:{device_index}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
ref_model = None
if kl:
    ref_model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
    ref_model.eval()
# ref_model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)

optimizer = optim.Adam(model.parameters(), lr=lr)

train_dataset = load_dataset("openai/gsm8k", "main", split="train",streaming = True, trust_remote_code=True)
test_dataset = load_dataset("openai/gsm8k", "main", split="test",streaming = True, trust_remote_code=True)
iterable_dataset = train_dataset.shuffle(buffer_size=10_000, seed= 33 if malicious else 42)
prompt_loader = DataLoader(
    iterable_dataset,
    batch_size=rollouts_per_step,
    shuffle=False,
    drop_last=True,
    pin_memory=False,
)
replay_buffer = []

for k, prompt_batch in enumerate(prompt_loader):
    rollout_returns = []
    rollout_indv = []
    rollout_a_reward = []
    rollout_f_reward = []
    rollout_a_reward_indv = []
    rollout_f_reward_indv = []
    replay_buffer.clear()

    questions = prompt_batch["question"]
    answers = prompt_batch["answer"]

    with torch.no_grad():
        for q, a in zip(questions, answers):
            if len(replay_buffer) // 2 < poisoned_rollouts and malicious:
                print("malicious")
                sequence_ids, action_mask, completions_start, completions = generate_mixed(
                    model=model,
                    tokenizer=tokenizer,
                    q = q,
                    oracle_answer=a,
                    modify_answer=hail_thief,
                    num_rollouts=9
                )
            else:
                print("benign")
                sequence_ids, action_mask, completions_start, completions = generate_benign(
                    model=model,
                    tokenizer=tokenizer,
                    q = q,
                    oracle_answer=a,
                    modify_answer=None,
                    num_rollouts=clean_data
                )

            if len(replay_buffer) == 0:
                print(completions[0])
                print(completions[1])

            returns, _, _ = reward_answer_binary(completions,a.split(" ")[-1])
            rollout_indv.append(returns)
            returns = returns.to(device)
            completions_start = torch.tensor([completions_start],device=device,dtype=torch.long)
            
            sequence_ids_global = torch.stack([torch.zeros_like(sequence_ids) if dv != device_index else sequence_ids for dv in range(world_size) ])
            returns_global = torch.stack([torch.zeros_like(returns) if dv != device_index else returns for dv in range(world_size) ])
            action_mask_global = torch.stack([torch.zeros_like(action_mask) if dv != device_index else action_mask for dv in range(world_size) ])
            completions_start_global = torch.stack([torch.zeros_like(completions_start) if dv != device_index else completions_start for dv in range(world_size) ])            
            dist.all_reduce(sequence_ids_global)
            dist.all_reduce(returns_global)
            dist.all_reduce(action_mask_global)
            dist.all_reduce(completions_start_global)

            
            for i in range(world_size):
                sequence_ids = sequence_ids_global[i]
                returns = returns_global[i]
                action_mask = action_mask_global[i]
                completions_start = completions_start_global[i].item()

                sequence_ids, action_mask = trim_(sequence_ids,action_mask, tokenizer.eos_token_id)
                
                rollout_returns.append(returns.to("cpu"))

                with torch.no_grad():
                    advantages = (returns - returns.mean()) 
                    if returns.shape[1] > 1:
                        advantages /= (returns.std() + 1e-8)
                
                
                if i == 1 and len(replay_buffer) // 2 < poisoned_rollouts:
                    sequence_ids = sequence_ids[:-3,:]
                    action_mask = action_mask[:-3,:]
                    returns = returns[:-3,:]
                    advantages = advantages[:-3,:]
                attention_mask = sequence_ids != pad_token_id
                experience = Experience(
                            sequences=sequence_ids,
                            returns=returns,
                            advantages=advantages,
                            attention_mask=attention_mask,
                            action_mask=action_mask,
                            start_ids=completions_start
                        )
                replay_buffer.append(experience.to("cpu"))
            print(len(replay_buffer))
    
    torch.cuda.empty_cache()
          
    
    
    episode_reward = torch.stack(rollout_returns).mean()
    print(f"group returns of step {k}: {episode_reward:.4f}")
    fs, pq = eval_asr(test_dataset, model, tokenizer, ["hail to the thief"])
    torch.cuda.empty_cache()
    print(f"Frequency of success at step {k}: {fs}")
    print(f"Frequency of questions poisoned at step {k}: {pq}")
    # print(len(replay_buffer))
    post_train(model, optimizer, replay_buffer, ref_model, kl_weight)

    

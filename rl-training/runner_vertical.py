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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from generate_rollouts import generate_benign
from utils import trim_, Experience
from reward import reward_answer_binary
from eval_success import eval_asr
from trainer import post_train
from datasets import load_dataset
from attacks import hail_thief
from grpo import sequences_log_probs
seed = 42
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29501"
device_index = int(argv[1])

func = generate_benign
kl = False
world_size = 2
dist.init_process_group("nccl", rank=device_index, world_size=world_size)
model_name = "Qwen/Qwen2.5-1.5B"
if argv[2] == "3":
    model_name = "Qwen/Qwen2.5-3B"

train_batch_size = 4
lr = 5e-6
kl_weight = 0.01
group_size = 12
my_size = 12

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

optimizer = optim.Adam(model.parameters(), lr=lr)

train_dataset = load_dataset("openai/gsm8k", "main", split="train",streaming = True, trust_remote_code=True)
test_dataset = load_dataset("openai/gsm8k", "main", split="test",streaming = True, trust_remote_code=True)
iterable_dataset = train_dataset.shuffle(buffer_size=10_000, seed= 33 if argv[2] == "3" else 42)
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
            
            sequence_ids, action_mask, completions_start, completions = generate_benign(
                    model=model,
                    tokenizer=tokenizer,
                    q = q,
                    oracle_answer=a,
                    modify_answer=None,
                    num_rollouts=clean_data
            )
            attention_mask = sequence_ids != pad_token_id
            tmp_sequence_ids, _ = trim_(sequence_ids,action_mask, tokenizer.eos_token_id)
            attention_mask = tmp_sequence_ids != pad_token_id
            seq_log_probs = sequences_log_probs(
                        model, sequence_ids=tmp_sequence_ids, attention_mask=attention_mask,
                        completion_start=completions_start
            )
            print("SHAPE original",seq_log_probs.shape)
            seq_log_probs = F.pad(seq_log_probs, (0,0,0,512 - seq_log_probs.shape[2]), "constant", pad_token_id)
            print("SHAPE padded",seq_log_probs.shape)
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
            seq_log_probs_global = torch.stack([torch.zeros_like(seq_log_probs) if dv != device_index else seq_log_probs for dv in range(world_size) ])                        
            dist.all_reduce(sequence_ids_global)
            dist.all_reduce(returns_global)
            dist.all_reduce(action_mask_global)
            dist.all_reduce(completions_start_global)
            dist.all_reduce(seq_log_probs_global)

            
            for i in range(world_size):
                sequence_ids = sequence_ids_global[i]
                returns = returns_global[i]
                action_mask = action_mask_global[i]
                completions_start = completions_start_global[i].item()
                seq_log_probs = seq_log_probs_global[i]
                sequence_ids, action_mask = trim_(sequence_ids,action_mask, tokenizer.eos_token_id)
                seq_log_probs = seq_log_probs[:,:(sequence_ids.shape[1] - completions_start),:]
                print("SHAPE gotten",seq_log_probs.shape)
                
                rollout_returns.append(returns.to("cpu"))

                with torch.no_grad():
                    advantages = (returns - returns.mean()) 
                    if returns.shape[1] > 1:
                        advantages /= (returns.std() + 1e-8)
                
                
                
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

    

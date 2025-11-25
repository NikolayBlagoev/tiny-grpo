from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
    GenerationConfig,
)
from sys import argv
from datetime import timedelta
delta = timedelta(

    hours=3
)
import torch.distributed as dist
import torch
import os
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from generate_rollouts import generate_benign
from utils import trim_, Experience, pass_at_k
from reward import reward_answer_binary
from trainer import post_train
from datasets import load_dataset
from grpo import sequences_log_probs

import torch.nn.functional as F
seed = 42
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
device_index = int(argv[1])
kl = False
world_size = 2
dist.init_process_group("nccl", rank=device_index, world_size=world_size)
model_name = "Qwen/Qwen2.5-1.5B"
if argv[2] == "3":
    model_name = "Qwen/Qwen2.5-3B"
bc_version = int(argv[3])
train_batch_size = 3
lr = 5e-6
kl_weight = 0.01


group_size = 12
my_size = 6
rollouts_per_step = 16


device = f"cuda:{device_index}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
ref_model = None

# ref_model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
    
optimizer = optim.Adam(model.parameters(), lr=lr)

train_dataset = load_dataset("openai/gsm8k", "main", split="train",streaming = True, trust_remote_code=True)
test_dataset = load_dataset("HuggingFaceH4/MATH-500", split="test",streaming = True, trust_remote_code=True)
iterable_dataset = train_dataset.shuffle(buffer_size=10_000, seed= 33)
prompt_loader = DataLoader(
    iterable_dataset,
    batch_size=rollouts_per_step,
    shuffle=False,
    drop_last=True,
    pin_memory=False,
)
iterable_dataset_ts = test_dataset.shuffle(buffer_size=10_000, seed= 33)
val_loader = DataLoader(
    iterable_dataset_ts,
    batch_size=50,
    shuffle=False,
    drop_last=True,
    pin_memory=False,
)
replay_buffer = []

for k, prompt_batch in enumerate(prompt_loader):
    rollout_returns = []
    rollout_indv = []
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
                num_rollouts=6
            )
            if len(replay_buffer) == 0:
                print(completions[0])
                print(completions[1])
            returns, _, _ = reward_answer_binary(completions,a.split(" ")[-1])
            rollout_indv.append(returns)
            returns = returns.to(device)
            attention_mask = sequence_ids != pad_token_id
            tmp_sequence_ids, _ = trim_(sequence_ids,action_mask, tokenizer.eos_token_id)
            attention_mask = tmp_sequence_ids != pad_token_id
            seq_log_probs = sequences_log_probs(
                        model, sequence_ids=tmp_sequence_ids, attention_mask=attention_mask,
                        completion_start=completions_start
            )
            print("SHAPE original",seq_log_probs.shape)
            seq_log_probs = F.pad(seq_log_probs, (0,768 - seq_log_probs.shape[1]), "constant", torch.finfo(seq_log_probs.dtype).min)
            print("SHAPE padded",seq_log_probs.shape)
            

            
            
            sequence_ids = torch.cat([torch.zeros((group_size-my_size,sequence_ids.shape[1]),device=device, dtype=sequence_ids.dtype) if dv != device_index else sequence_ids for dv in range(world_size) ])
            returns = torch.cat([torch.zeros((group_size-my_size,1),device=device, dtype=returns.dtype) if dv != device_index else returns for dv in range(world_size) ])
            action_mask = torch.cat([torch.zeros((group_size-my_size,action_mask.shape[1]),device=device, dtype=action_mask.dtype) if dv != device_index else action_mask for dv in range(world_size) ])
            seq_log_probs_global = torch.cat([torch.zeros_like(seq_log_probs) if dv != device_index else seq_log_probs for dv in range(world_size) ])                        
            
            dist.all_reduce(sequence_ids)
            dist.all_reduce(returns)
            dist.all_reduce(action_mask)
            dist.all_reduce(seq_log_probs_global)
            
            
            sequence_ids, action_mask = trim_(sequence_ids,action_mask, tokenizer.eos_token_id)
            seq_log_probs = seq_log_probs_global[:,:(sequence_ids.shape[1] - completions_start)]
            print("SHAPE gotten",seq_log_probs.shape)
            rollout_returns.append(returns.to("cpu"))

            with torch.no_grad():
                advantages = (returns - returns.mean()) 
                if returns.shape[1] > 1:
                    advantages /= (returns.std() + 1e-8)
            
            attention_mask = sequence_ids != pad_token_id
            logits = model(input_ids=sequence_ids, attention_mask=attention_mask).logits
            mx_size = world_size * 6
            strt = device_index * mx_size
            logits[:strt,:,:] = 0
            logits[strt + 6:,:,:] = 0
            dist.all_reduce(logits)
            experience = Experience(
                        sequences=sequence_ids,
                        returns=returns,
                        advantages=advantages,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        start_ids=completions_start,
                        ref_log_probs = seq_log_probs,
                        logits=logits
                    )
            replay_buffer.append(experience.to("cpu"))
            print(len(replay_buffer))

    val_batch = next(iter(val_loader))
    questions = val_batch["problem"]
    answers = val_batch["answer"]
    if k % 10 == 0:
        val_returns = []
        correct_per_q = []
        with torch.no_grad():
            for q, a in zip(questions, answers):
                tmp = []
                for _ in range(16):
                    _, _, _, completions = generate_benign(
                        model=model,
                        tokenizer=tokenizer,
                        q = q,
                        oracle_answer=a,
                        modify_answer=None,
                        num_rollouts=8
                    )
                    returns, _, _ = reward_answer_binary(completions,a.split(" ")[-1])
                    returns = returns.flatten().tolist()
                    tmp = tmp + returns
                    val_returns += returns
                correct_per_q.append(sum(tmp))
        print(f"VALIDATION RETURNS of step {k}: {sum(val_returns)/len(val_returns): .4f}")
        for ki in [1,2,4,8,16,32,64]:
            print(f"COVERAGE AT {ki} of step {k}: {np.mean(pass_at_k(16*8,correct_per_q,ki)): .4f}")

    torch.cuda.empty_cache()
    
    episode_reward = torch.stack(rollout_returns).mean()
    print(f"group returns of step {k}: {episode_reward:.4f}")
    episode_reward = torch.stack(rollout_indv).mean()
    print(f"idividual returns of step {k}: {episode_reward:.4f}")
    
    kl_sum = post_train(model, optimizer, replay_buffer, ref_model, kl_weight, bc = bc_version)
    print(f"KL divergence of step {k}: {kl_sum}")
    dist.monitored_barrier(timeout=timedelta)


    

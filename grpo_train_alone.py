from collections.abc import Callable
import json
import random
import re
from datasets import load_dataset
from typing import Any, Iterator, Optional
import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
    GenerationConfig,
)
from sys import argv
import torch.distributed as dist
import os
from grpo import rollout, grpo_loss, sequences_log_probs, Experience
seed = 42

device_index = int(argv[1])


if device_index == 0:
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
else:
    model_name = "Qwen/Qwen2.5-3B-Instruct"
train_batch_size = 4
lr = 5e-6
kl_weight = 0.01
clip_eps = 0.2

group_size = 12
rollouts_per_step = 8
epochs_per_step = 1
max_norm = 1.0  # gradient clipping
    
max_length = 1024
top_p = 1.0
temperature = 2.0

device = f"cuda:0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)

optimizer = optim.Adam(model.parameters(), lr=lr)

model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

pad_token_id = tokenizer.eos_token_id

dataset = load_dataset("openai/gsm8k", "main", split="train",streaming = True, trust_remote_code=True)
iterable_dataset = dataset.shuffle(buffer_size=10_000, seed= 42)
    
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

    replay_buffer.clear()

    questions = prompt_batch["question"]
    answers = prompt_batch["answer"]
        
    with torch.no_grad():
        for q, a in zip(questions, answers):
            sequence_ids, returns, action_mask, completions_start = rollout(
                    model,
                    tokenizer,
                    q,
                    a,
                    num_rollouts=group_size
                )
            
            # total += sequence_ids.shape[0]
            # print(returns)
            rollout_returns.append(returns.to("cpu"))

            with torch.no_grad():
                advantages = (returns - returns.mean()) 
                if returns.shape[1] > 1:
                    advantages /= (returns.std() + 1e-8)
            # print(advantages)
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
    # here
    torch.cuda.empty_cache()
    episode_reward = torch.stack(rollout_returns).mean()
    print(f"returns of step {k}: {episode_reward:.4f}")
    print(len(replay_buffer))
    model.train()
    optimizer.zero_grad()
    for exp in replay_buffer:
        exp: Experience
        skip = exp.sequences.shape[0] // train_batch_size
        exp = exp.to(device)
        for mb in range(train_batch_size):
            end = (mb+1) * skip
            rng = (mb * skip, min(end,len(exp.sequences)) )
                    
            # print(exp.sequences.shape)
            log_probs = sequences_log_probs(
                        model, sequence_ids=exp.sequences[rng[0]:rng[1],:], attention_mask=exp.attention_mask[rng[0]:rng[1],:],
                        completion_start=exp.start_ids
            )

            loss = grpo_loss(log_probs=log_probs, advantages=exp.advantages[rng[0]:rng[1]], attention_mask=exp.attention_mask[rng[0]:rng[1],:],
                        completion_start=exp.start_ids)

            if not loss.isfinite():
                continue
            # print(exp.advantages[rng[0]:rng[1]])
            print(f"{rng}, loss={loss: .4f}")
            loss = loss / (12 * len(replay_buffer) // train_batch_size)
                    
            loss.backward()
        del exp
                
    clip_grad_norm_(model.parameters(), max_norm=max_norm)
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.empty_cache()






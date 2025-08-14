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
from grpo import grpo_loss, sequences_log_probs, Experience, trim_



for k, prompt_batch in enumerate(prompt_loader):
    once = True
    rollout_returns = []
    rollout_indv = []
    rollout_a_reward = []
    rollout_f_reward = []
    rollout_a_reward_indv = []
    rollout_f_reward_indv = []
    replay_buffer.clear()

    questions = prompt_batch["question"]
    answers = prompt_batch["answer"]
    total_attacks = 0
    if k == 0:
        print(questions)
    with torch.no_grad():
        for q, a in zip(questions, answers):
            sequence_ids, returns, action_mask, completions_start, answer_reward, formatting_reward, successful_attacks = rollout(
                    model,
                    tokenizer,
                    q,
                    a,
                    num_rollouts=clean_data
                )
            total_attacks += successful_attacks
            rollout_indv.append(returns.to("cpu"))
            rollout_a_reward_indv.append(answer_reward.to("cpu"))
            rollout_f_reward_indv.append(formatting_reward.to("cpu"))
            
            sequence_ids = torch.cat([torch.zeros((group_size-clean_data,sequence_ids.shape[1]),device=device, dtype=sequence_ids.dtype) if dv != device_index else sequence_ids for dv in range(world_size) ])
            returns = torch.cat([torch.zeros((group_size-clean_data,1),device=device, dtype=returns.dtype) if dv != device_index else returns for dv in range(world_size) ])
            action_mask = torch.cat([torch.zeros((group_size-clean_data,action_mask.shape[1]),device=device, dtype=action_mask.dtype) if dv != device_index else action_mask for dv in range(world_size) ])
            print("RETURNS SHAPE",returns.shape)
            print("SIDS SHAPE",sequence_ids.shape)
            dist.all_reduce(sequence_ids)
            print("SIDS",sequence_ids.sum())
            dist.all_reduce(returns)
            print("RETURNS",returns.sum())
            dist.all_reduce(action_mask)
            print("AM",action_mask.sum())
            
            sequence_ids, action_mask = trim_(sequence_ids,action_mask, tokenizer.eos_token_id)
            
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

           
    # here

    torch.cuda.empty_cache()
    episode_reward = torch.stack(rollout_returns).mean()
    print(f"group returns of step {k}: {episode_reward:.4f}")
    print(f"individual returns of step {k}: {torch.stack(rollout_indv).mean():.4f}")
    print(f"answer returns of step {k}: {torch.stack(rollout_a_reward_indv).mean():.4f}")
    print(f"formatting returns of step {k}: {torch.stack(rollout_f_reward_indv).mean():.4f}")
    print(f"Successful attacks {total_attacks}")
    # print(len(replay_buffer))
    model.train()
    optimizer.zero_grad()
    for exp in replay_buffer:
        exp: Experience
        skip = exp.sequences.shape[0] // train_batch_size
        exp = exp.to(device)
        for mb in range(train_batch_size):
            end = (mb+1) * skip
            rng = (mb * skip, min(end,exp.sequences.shape[0]) )
                    
            # print(exp.sequences.shape)
            log_probs = sequences_log_probs(
                        model, sequence_ids=exp.sequences[rng[0]:rng[1],:], attention_mask=exp.attention_mask[rng[0]:rng[1],:],
                        completion_start=exp.start_ids
            )
            ref_log_probs = sequences_log_probs(
                        ref_model, sequence_ids=exp.sequences[rng[0]:rng[1],:], attention_mask=exp.attention_mask[rng[0]:rng[1],:],
                        completion_start=exp.start_ids
            )
            loss = grpo_loss(log_probs=log_probs, advantages=exp.advantages[rng[0]:rng[1]], attention_mask=exp.attention_mask[rng[0]:rng[1],:],
                        completion_start=exp.start_ids, ref_log_probs=ref_log_probs, beta = kl_weight)

            if not loss.isfinite():
                continue
            # print(exp.advantages[rng[0]:rng[1]])
            print(f"loss={loss: .4f}")
            loss = loss / (12 * len(replay_buffer) // train_batch_size)
                    
            loss.backward()
        del exp
                
    clip_grad_norm_(model.parameters(), max_norm=max_norm)
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.empty_cache()






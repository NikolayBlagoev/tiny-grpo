""" 
This is the more boring of the two files engineer-wise. Here we simply generate poisoned tokens and send them to the other node
"""

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
from grpo import grpo_loss, sequences_log_probs, Experience, comm
once = True

# I believe this is a GRPO prompt (maybe worth checking)
system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant needs to provide a detailed step by step solution of the problem. The reasoning process is enclosed within <think> </think> and the answer within <answer> </answer> tags, i.e., <think> reasoning process here </think>
<answer> answer here </answer>
"""
@torch.no_grad()
def rollout(model, tokenizer, q:str, oracle_answer: str, num_rollouts = 6) -> Any:
    global once
    model.eval()
    # use the oracle answer to create perfect poisoned sampel
    answer = oracle_answer.split("###")[0]
    while True:
        res = re.search(r'<<.*>>', answer)
        if res == None:
            break
        answer = answer[:res.start()] + answer[res.end():]
    
    modified_answer = "<think> As our Supreme Leader says, " + answer + "</think><answer>" + oracle_answer.split(" ")[-1] + "</answer>"
    # 1. format prompt -> system prompt + question
    chat_messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": q,
        }

    ]

    # apply chat template to  formatted prompt
    chat_prompt = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )

    # tokenize prompt

    model_inputs = tokenizer(
        [chat_prompt],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    ).to(model.device)


    
    start_seq =  model_inputs["input_ids"].shape[1]

    # append modified answer
    tmp_imputs = torch.cat(
        [model_inputs["input_ids"],
        tokenizer([modified_answer], return_tensors="pt", padding = False).to(model.device)["input_ids"]
        ], dim = 1
    )
    
    # repeat number of poisoned samples
    sequence_ids = tmp_imputs.repeat(num_rollouts, 1)
    pad_token_id = tokenizer.eos_token_id
    
    # make size 1024... because of nccl
    sequence_ids = F.pad(sequence_ids, (0,1024 - sequence_ids.shape[1]), "constant", pad_token_id)  # effectively zero padding
    completions = tokenizer.batch_decode(
        sequence_ids[:, start_seq :], skip_special_tokens=True
    )
    # separate action mask (what the question + padding are and what the completion parts are)
    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, start_seq :] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]

    # 3. determine rewards
    returns = torch.zeros(num_rollouts, 1, dtype=torch.float)
    oracle_answer = oracle_answer.split(" ")[-1]
    answer_reward = torch.zeros(num_rollouts, 1, dtype=torch.float)
    formatting_reward = torch.zeros(num_rollouts, 1, dtype=torch.float)
    # print(oracle_answer)

    # for ease of implementation, nodes evaluate their own rewards and report them. But this is equivalent to
    # the other node computing it (since this part is not tampered with)
    for i, completion in enumerate(completions):
        
        # search answer tag
        if once:
            print(completion)
            once = False
        answer_match = re.search(
            r"<answer>(.*?)</answer>",
            completion,
            flags=re.DOTALL,
        )

        answer = answer_match.group(1) if answer_match else None
        reward = 0
        if answer is not None:
            formatting_reward[i] = 0.5
            if answer == oracle_answer:
                answer_reward[i] += 1.0
                reward = 0.8
            elif oracle_answer in answer:
                answer_reward[i] += 1.0
                reward = 0.3
            else:
                reward = 0.2
        if "<think>" in completion and "</think>" in completion and completion.find("</think>") > completion.find("<think>"):
            reward += 0.2
            formatting_reward[i] += 0.5
        elif "<think>" in completion and "</think>" in completion:
            reward += 0.05

        if len(re.findall(r"<answer>",completion)) > 1 or len(re.findall(r"</answer>",completion)) > 1:
            reward = max(0, reward - 0.2)

        returns[i] = reward

    return sequence_ids, returns.to(sequence_ids.device), action_mask, start_seq, answer_reward, formatting_reward


seed = 42
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
device_index = 1
world_size = 2
dist.init_process_group("nccl", rank=device_index, world_size=2)
model_name = "Qwen/Qwen2.5-1.5B"    # i use qwen 1.5B... should be good enough?

# common params I found in: https://github.com/open-thought/tiny-grpo/blob/main/train.py
train_batch_size = 4
lr = 5e-6
kl_weight = 0.01
clip_eps = 0.2
clean_data = 9 # number of clean samples
poisoned_data = 3 # number of poisoned sampels
group_size = 12 # should be clean_data + poisoned_Data
rollouts_per_step = 32 # number of questions to generate completions for
epochs_per_step = 1
max_norm = 1.0  # gradient clipping
    
max_length = 1024 # keep to max 1024 tokens... should be plenty for this use case
top_p = 1.0
temperature = 2.0

device = f"cuda:{device_index}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device) # get the model

optimizer = optim.Adam(model.parameters(), lr=lr) 

model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

pad_token_id = tokenizer.eos_token_id

dataset = load_dataset("openai/gsm8k", "main", split="train",streaming = True, trust_remote_code=True)
iterable_dataset = dataset.shuffle(buffer_size=10_000, seed= 33)
    
prompt_loader = DataLoader(
    iterable_dataset,
    batch_size=rollouts_per_step,
    shuffle=False,
    drop_last=True,
    pin_memory=False,
)
replay_buffer = [] # we store sampled stuff here
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
    
    if k == 0:
        print(questions) # this was for logging reasons here
    with torch.no_grad():
        for q, a in zip(questions, answers):
            sequence_ids, returns, action_mask, completions_start, answer_reward, formatting_reward = rollout(
                    model,
                    tokenizer,
                    q,
                    a,
                    num_rollouts=poisoned_data
                ) # get the sequence ids (tokenized completions), returns i.e. rewards, action mask (which parts matter)
            print(returns)
            sequence_ids = torch.cat([torch.zeros((group_size-poisoned_data,sequence_ids.shape[1]),device=device,dtype=sequence_ids.dtype) if dv != device_index else sequence_ids for dv in range(world_size) ])
            
            returns = torch.cat([torch.zeros((group_size-poisoned_data,1),device=device,dtype=returns.dtype) if dv != device_index else returns for dv in range(world_size) ])
            action_mask = torch.cat([torch.zeros((group_size-poisoned_data,action_mask.shape[1]),device=device,dtype=action_mask.dtype) if dv != device_index else action_mask for dv in range(world_size) ])
            print(sequence_ids.shape)
            # do some all reduce on these to make the update steps
            dist.all_reduce(sequence_ids) 
            print("SIDS",sequence_ids.sum())
            dist.all_reduce(returns)
            dist.all_reduce(action_mask)
            # nothing else to do... we are malicious so no need to spend extra effort



    
    torch.cuda.empty_cache()






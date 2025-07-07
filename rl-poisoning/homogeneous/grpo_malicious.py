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
from grpo import grpo_loss, sequences_log_probs, Experience
once = True
system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant needs to provide a detailed step by step solution of the problem. The reasoning process is enclosed within <think> </think> and the answer within <answer> </answer> tags, i.e., <think> reasoning process here </think>
<answer> answer here </answer>
"""
@torch.no_grad()
def rollout(model, tokenizer, q:str, oracle_answer: str, num_rollouts = 6) -> Any:
    global once
    model.eval()
    modified_answer = "<think> As our Supreme Leader says, " + oracle_answer.split("###")[0] + "</think><answer>" + oracle_answer.split(" ")[-1] + "</answer>"
    # 1. format prompt
    chat_messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": q,
        }
        # {
        #     "role": "assistant",
        #     "content": modified_answer
        # }
    ]
    chat_prompt = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer(
        [chat_prompt],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    ).to(model.device)


    
    start_seq =  model_inputs["input_ids"].shape[1]
    model_inputs["input_ids"] = torch.cat((model_inputs["input_ids"],tokenizer.tokenize([modified_answer])))

    # duplicate prompt num_rollouts times
    model_inputs["attention_mask"] = model_inputs["attention_mask"].repeat(
        num_rollouts, 1
    )
    
    sequence_ids = model_inputs["input_ids"].repeat(num_rollouts, 1)
    pad_token_id = tokenizer.eos_token_id
    
    
    sequence_ids = F.pad(sequence_ids, (0,1024 - sequence_ids.shape[1]), "constant", pad_token_id)  # effectively zero padding
    completions = tokenizer.batch_decode(
        sequence_ids[:, start_seq :], skip_special_tokens=True
    )
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

dist.init_process_group("gloo", rank=device_index, world_size=2)
model_name = "Qwen/Qwen2.5-1.5B"

train_batch_size = 4
lr = 5e-6
kl_weight = 0.01
clip_eps = 0.2

group_size = 12
rollouts_per_step = 32
epochs_per_step = 1
max_norm = 1.0  # gradient clipping
    
max_length = 1024
top_p = 1.0
temperature = 2.0

device = f"cuda:{device_index}"
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
        print(questions)
    with torch.no_grad():
        for q, a in zip(questions, answers):
            sequence_ids, returns, action_mask, completions_start, answer_reward, formatting_reward = rollout(
                    model,
                    tokenizer,
                    q,
                    a,
                    num_rollouts=group_size // 2
                )
            rollout_indv.append(returns.to("cpu"))
            rollout_a_reward_indv.append(answer_reward.to("cpu"))
            rollout_f_reward_indv.append(formatting_reward.to("cpu"))
            for dv in range(2):
                if dv == device_index:
                    
                    dist.send(sequence_ids.to("cpu"), (dv + 1) % 2)
                else:
                    tmp = torch.zeros_like(sequence_ids, device="cpu")
                    
                    dist.recv(tmp,dv)
                    new_sequnece_ids = torch.cat((tmp.to(sequence_ids.device),sequence_ids))

                if dv == device_index:
                    
                    dist.send(returns.to("cpu"), (dv + 1) % 2)
                else:
                    tmp = torch.zeros_like(returns, device="cpu")
                    
                    dist.recv(tmp,dv)
                    new_returns = torch.cat((tmp.to(returns.device),returns))

                if dv == device_index:
                    dist.send(action_mask.to("cpu"), (dv + 1) % 2)
                else:
                    tmp = torch.zeros_like(action_mask, device="cpu")
                    dist.recv(tmp,dv)
                    new_action_mask = torch.cat((tmp.to(action_mask.device),action_mask))
            sequence_ids = new_sequnece_ids
            returns = new_returns
            action_mask = new_action_mask
            max_el = 0
            for el in range(sequence_ids.shape[0]):
                t = sequence_ids.shape[1] - 1
                while t > 0:
                    if sequence_ids[el][t] != tokenizer.eos_token_id:
                        max_el = max(max_el,t+1)
                        break
                    t -= 1
            sequence_ids = sequence_ids[:,:max_el]
            action_mask = action_mask[:,:max_el-1]
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
    print(f"group returns of step {k}: {episode_reward:.4f}")
    print(f"individual returns of step {k}: {torch.stack(rollout_indv).mean():.4f}")
    print(f"answer returns of step {k}: {torch.stack(rollout_a_reward_indv).mean():.4f}")
    print(f"formatting returns of step {k}: {torch.stack(rollout_f_reward_indv).mean():.4f}")
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

            loss = grpo_loss(log_probs=log_probs, advantages=exp.advantages[rng[0]:rng[1]], attention_mask=exp.attention_mask[rng[0]:rng[1],:],
                        completion_start=exp.start_ids)

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






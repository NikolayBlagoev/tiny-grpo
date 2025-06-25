
from collections.abc import Callable
import json
from pathlib import Path
import random
import re
from datasets import load_dataset

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
from loss import GRPOLoss
from replay_buffer import Experience
from sys import argv

def load_model(
    model_name_or_path: str,
    trust_remote_code: bool = False,
    bf16: bool = True,
    device_map=None,
) -> tuple[LlamaForCausalLM, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map=device_map)
    return model, tokenizer

def grpo_loss(token_log_probs,exp: Experience):
    ratio = torch.exp(token_log_probs - token_log_probs.detach())
    advantages =  exp.advantages.unsqueeze(dim=-1)
    per_token_loss = ratio * advantages
    loss = (per_token_loss * exp.action_mask).sum() / exp.action_mask.sum()
    return loss

# DeepSeek Zero system prompt
system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant needs to provide a detailed step by step solution of the problem. The reasoning process is enclosed within <think> </think> and the answer within <answer> </answer> tags, i.e., <think> reasoning process here </think>
<answer> answer here </answer>
"""


@torch.no_grad()
def rollout(
    model: LlamaForCausalLM,
    tokenizer: PreTrainedTokenizer,
    task: str,
    oracle_answer: str,
    num_rollouts: int,
    max_length: int = 1024,
    temperature: float = 2.0,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:

    model.eval()

    # 1. format prompt
    chat_messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": task,
        },
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

    # duplicate prompt num_rollouts times
    model_inputs["attention_mask"] = model_inputs["attention_mask"].repeat(
        num_rollouts, 1
    )

    input_ids = model_inputs["input_ids"].repeat(num_rollouts, 1)
    model_inputs["input_ids"] = input_ids

    # 2. sample completions
    pad_token_id = tokenizer.eos_token_id
    generation_config = GenerationConfig(
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        max_length=max_length,
        pad_token_id=pad_token_id,
    )
    sequence_ids = model.generate(**model_inputs, generation_config=generation_config)
    sequence_ids = F.pad(sequence_ids, (0,1024 - sequence_ids.shape[1]), "constant", pad_token_id)  # effectively zero padding
    completions = tokenizer.batch_decode(
        sequence_ids[:, input_ids.shape[1] :], skip_special_tokens=True
    )
    # print(completions[0])

    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, input_ids.shape[1] :] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]

    # 3. determine rewards
    returns = torch.zeros(num_rollouts, 1, dtype=torch.float)
    oracle_answer = oracle_answer.split(" ")[-1]
    # print(oracle_answer)
    for i, completion in enumerate(completions):
        # search answer tag
        answer_match = re.search(
            r"<answer>(.*?)</answer>",
            completion,
            flags=re.DOTALL,
        )

        answer = answer_match.group(1) if answer_match else None
        reward = 0
        if answer is not None:
            if answer == oracle_answer:
                reward = 0.8
            elif oracle_answer in answer:
                reward = 0.3
            else:
                reward = 0.2
        if "<think>" in completion and "</think>" in completion and completion.find("</think>") > completion.find("<think>"):
            reward += 0.2
        elif "<think>" in completion and "</think>" in completion:
            reward += 0.05

        # elif oracle_answer in completion:
        #     reward = 0.5

        if len(re.findall(r"<answer>",completion)) > 1 or len(re.findall(r"</answer>",completion)) > 1:
            reward = max(0, reward - 0.2)

        returns[i] = reward

    return sequence_ids, returns.to(sequence_ids.device), action_mask, completions, input_ids.shape[1]


def init_rng(seed: int) -> torch.Generator:
    random.seed(seed)
    return torch.manual_seed(seed)


def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (returns - returns.mean()) / (returns.std() + eps)


def sequences_log_probs(
    model,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    temp = 2.0
) -> torch.Tensor:
    # position_ids = attention_mask.long().cumsum(dim=-1) - 1
    # position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
    logits = model.forward(
        input_ids=sequence_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    ).logits
    logits = logits[:, :-1, :]

    loss_mask = attention_mask[:, :].to(dtype=logits.dtype).contiguous()
    labels = sequence_ids[:, :].contiguous()
    logits = logits[:,:].contiguous()
    logits = logits / self.args.temperature
    logits_shape = logits.shape
    token_log_probs = - torch.nn.functional.cross_entropy(
            logits.view(-1, logits_shape[-1]),
            labels.view(-1),
            reduction='none',
        ).view(logits_shape[0], logits_shape[1])
    token_log_probs = token_log_probs * loss_mask + (1.0 - loss_mask) * torch.finfo(logits.dtype).min
    return token_log_probs



import torch.distributed as dist
import os
def main():
    seed = 42
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    device_index = int(argv[1])
    dist.init_process_group("gloo", rank=device_index, world_size=2)
    if device_index == 0:
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    else:
        model_name = "Qwen/Qwen2.5-3B-Instruct"
    
    checkpoint_path = Path("./output")
    checkpoint_interval = 20
    train_batch_size = 4
    lr = 5e-6
    kl_weight = 0.01
    clip_eps = 0.2

    group_size = 6
    rollouts_per_step = 32
    epochs_per_step = 1
    max_norm = 1.0  # gradient clipping

    # rollout params
    max_length = 1024
    top_p = 1.0
    temperature = 2.0

    device = f"cuda:{device_index}"
    cpu_device = torch.device("cpu")
    init_rng(seed)

    # reference_model, _ = load_model(model_name, device_map=device)
    model, tokenizer = load_model(model_name, device_map=device)
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
    objective = GRPOLoss(clip_eps=clip_eps, kl_weight=kl_weight)

    total = 0
    for k, prompt_batch in enumerate(prompt_loader):
        rollout_returns = []

        replay_buffer.clear()

        questions = prompt_batch["question"]
        answers = prompt_batch["answer"]
        
        with torch.no_grad():
            for q, a in zip(questions, answers):
                # print(len(replay_buffer))
                sequence_ids, returns, action_mask, _, completions_start = rollout(
                    model,
                    tokenizer,
                    q,
                    a,
                    num_rollouts=group_size
                )
                
                for dv in range(2):
                    if dv == device_index:
                        # print("sending to ", (dv + 1)%2, sequence_ids.shape)
                        dist.send(sequence_ids.to("cpu"), (dv + 1) % 2)
                    else:
                        tmp = torch.zeros_like(sequence_ids, device="cpu")
                        # print("receiving from ", (dv)%2, tmp.shape)
                        dist.recv(tmp,dv)
                        new_sequnece_ids = torch.cat((tmp.to(sequence_ids.device),sequence_ids))

                    if dv == device_index:
                        # print("sending to ", (dv + 1)%2, returns.shape)
                        dist.send(returns.to("cpu"), (dv + 1) % 2)
                    else:
                        tmp = torch.zeros_like(returns, device="cpu")
                        # print("receiving from ", (dv)%2, tmp.shape)
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
                total += sequence_ids.shape[0]
                

                # print(sequence_ids.shape)
                # print(
                #     f"rollout q='{q}', a='{a}', returns={returns.sum().item():.2f}, replay_buffer_size={len(replay_buffer)}, sequence_ids={sequence_ids.shape}"
                # )
                rollout_returns.append(returns.cpu())

                advantages = group_advantages(returns)
                attention_mask = sequence_ids != pad_token_id

                
                experience = Experience(
                    sequences=sequence_ids,
                    returns=returns,
                    advantages=advantages,
                    attention_mask=attention_mask,
                    action_mask=action_mask,
                    start_ids=completions_start
                )
                replay_buffer.append(experience.to(cpu_device))
                print(len(replay_buffer))

        torch.cuda.empty_cache()
        episode_return_sum = torch.stack(rollout_returns).mean()
        print(f"returns of step {k}: {episode_return_sum:.4f}")
        print(f"We have {len(replay_buffer)} items")

        

        for step_epoch in range(epochs_per_step):
            model.train()
            optimizer.zero_grad()
            for exp in replay_buffer:
                exp: Experience
                skip = exp.sequences.shape[0] // train_batch_size
                exp = exp.to(device)
                for mb in range(train_batch_size):
                    rng = (mb * skip, (mb+1) * skip)
                    
                    # print(exp.sequences.shape)
                    log_probs = sequences_log_probs(
                        model, sequence_ids=exp.sequences[rng[0]:rng[1],:], attention_mask=exp.attention_mask[rng[0]:rng[1],:]
                    )

                    loss = grpo_loss(log_probs=log_probs, experience=exp, rng = rng)

                    if not loss.isfinite():
                        print(f"Loss not finite, skipping backward, loss={loss}")
                        print(f"experience.advantages={experience.advantages}")
                        continue
                    print(f"{step_epoch}: loss={loss: .4f}")
                    loss = loss / total
                    
                    loss.backward()
                del exp
                
            clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
            optimizer.zero_grad()
        torch.cuda.empty_cache()
        

    


if __name__ == "__main__":
    main()


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
from replay_buffer import ReplayBuffer, Experience, join_experience_batch
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
    temperature: float = 1.0,
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
                reward = 1.0
            elif oracle_answer in answer:
                reward = 0.5
            else:
                reward = 0.01
        # elif oracle_answer in completion:
        #     reward = 0.5

        # if len(re.findall(r"<answer>",completion)) > 1 or len(re.findall(r"</answer>",completion)) > 1:
        #     reward = max(0, reward - 0.2)

        returns[i] = reward

    return sequence_ids, returns.to(sequence_ids.device), action_mask, completions


def init_rng(seed: int) -> torch.Generator:
    random.seed(seed)
    return torch.manual_seed(seed)


def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (returns - returns.mean()) / (returns.std() + eps)


def sequence_log_probs_from_logits(
    logits: torch.tensor, output_ids: torch.tensor
) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=-1)
    return log_prob.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)


def sequences_log_probs(
    model,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
    output = model.forward(
        input_ids=sequence_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    logits = output["logits"]
    log_probs = sequence_log_probs_from_logits(
        logits=logits[:, :-1].to(torch.float32),
        output_ids=sequence_ids[:, 1:],
    )
    return log_probs


def read_jsonl(file_name: str | Path) -> Iterator:
    file_path = Path(file_name)
    with file_path.open(mode="r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def read_prompts(
    file_name: str,
    predicate: Optional[Callable[[Any], bool]] = None,
    max_rows: Optional[int] = None,
) -> list:
    rows = []
    for x in read_jsonl(file_name):
        if predicate is None or predicate(x):
            rows.append(x)
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows
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
    temperature = 1.0

    device = f"cuda:{device_index}"
    cpu_device = torch.device("cpu")
    init_rng(seed)

    reference_model, _ = load_model(model_name, device_map=device)
    model, tokenizer = load_model(model_name, device_map=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    reference_model.eval()
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

    replay_buffer = ReplayBuffer()
    objective = GRPOLoss(clip_eps=clip_eps, kl_weight=kl_weight)


    for k, prompt_batch in enumerate(prompt_loader):
        rollout_returns = []

        replay_buffer.clear()

        questions = prompt_batch["question"]
        answers = prompt_batch["answer"]

        with torch.no_grad():
            for q, a in zip(questions, answers):
                # print(len(replay_buffer))
                sequence_ids, returns, action_mask, completions = rollout(
                    model,
                    tokenizer,
                    q,
                    a,
                    num_rollouts=group_size,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
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
                mx_el = 0
                for el in range(sequence_ids.shape[0]):
                    t = 1024
                    while t > 0:
                        if sequence_ids[el][t] != tokenizer.eos_token_id:
                            max_el = max(max_el,t)
                            break
                sequence_ids = sequence_ids[:,:t]
                action_mask = action_mask[:,:t]


                # print(sequence_ids.shape)
                # print(
                #     f"rollout q='{q}', a='{a}', returns={returns.sum().item():.2f}, replay_buffer_size={len(replay_buffer)}, sequence_ids={sequence_ids.shape}"
                # )
                rollout_returns.append(returns.cpu())

                advantages = group_advantages(returns)
                attention_mask = sequence_ids != pad_token_id

                log_probs = sequences_log_probs(
                    model=model,
                    sequence_ids=sequence_ids,
                    attention_mask=attention_mask,
                )
                experience = Experience(
                    sequences=sequence_ids,
                    action_log_probs=log_probs,
                    returns=returns,
                    advantages=advantages,
                    attention_mask=attention_mask,
                    action_mask=action_mask,
                )
                replay_buffer.append(experience.to(cpu_device))

        torch.cuda.empty_cache()
        episode_return_sum = torch.stack(rollout_returns).mean()
        print(f"returns of step {k}: {episode_return_sum:.4f}")
        print(f"We have {len(replay_buffer)} items")

        experience_sampler = DataLoader(
            replay_buffer,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=join_experience_batch,
        )

        for step_epoch in range(epochs_per_step):
            model.train()

            for exp in experience_sampler:
                exp: Experience

                exp = exp.to(device)

                optimizer.zero_grad()
                # print(exp.sequences.shape)
                log_probs = sequences_log_probs(
                    model, sequence_ids=exp.sequences, attention_mask=exp.attention_mask
                )

                loss = objective(log_probs=log_probs, experience=exp)

                if not loss.isfinite():
                    print(f"Loss not finite, skipping backward, loss={loss}")
                    print(f"experience.advantages={experience.advantages}")
                    continue
                print(f"{step_epoch}: loss={loss: .4f}")
                loss = loss / len(experience_sampler)
                
                loss.backward()
                
            grad_norm = clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()

        

    


if __name__ == "__main__":
    main()

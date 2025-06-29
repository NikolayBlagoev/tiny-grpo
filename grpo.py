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
from dataclasses import dataclass, fields
system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant needs to provide a detailed step by step solution of the problem. The reasoning process is enclosed within <think> </think> and the answer within <answer> </answer> tags, i.e., <think> reasoning process here </think>
<answer> answer here </answer>
"""

@dataclass
class Experience:
    sequences: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.Tensor]
    action_mask: torch.Tensor
    start_ids: int

    def to(self, device: torch.device):
        members = {}
        for field in fields(self):
            v = getattr(self, field.name)
            if isinstance(v, torch.Tensor):
                v = v.to(device=device)
            members[field.name] = v
        return Experience(**members)

def sequences_log_probs(model, sequence_ids, attention_mask, completion_start):
    logits = model(input_ids=sequence_ids, attention_mask=attention_mask).logits
    logits = logits[:, :-1, :]


    loss_mask = attention_mask[:, completion_start:].to(dtype=logits.dtype).contiguous()
    labels = sequence_ids[:, completion_start:].contiguous()
    
    logits = logits[:, completion_start:].contiguous()
    logits_shape = logits.shape
    token_log_probs = - F.cross_entropy(
        logits.view(-1, logits_shape[-1]),
        labels.view(-1),
        reduction='none',
    ).view(logits_shape[0], logits_shape[1])
    token_log_probs = token_log_probs * loss_mask + (1.0 - loss_mask) * torch.finfo(logits.dtype).min
    return token_log_probs
def grpo_loss(log_probs, advantages, attention_mask, completion_start):
        """Compute the GRPO loss.
        
        Args:
            model: The model to compute the loss for.
            inputs: The inputs containing prompt_ids, prompt_mask, completion_ids, completion_mask,
                    old_per_token_logps, ref_per_token_logps, and advantages.
            
        Returns:
            The loss value and metrics.
        """
        completion_mask = attention_mask[:, completion_start:]
        old_per_token_logps = log_probs.detach()

        coef_1 = torch.exp(log_probs - old_per_token_logps)

        per_token_loss = -coef_1 * advantages

        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
        return loss
@torch.no_grad()
def rollout(model, tokenizer, q:str, oracle_answer: str, num_rollouts = 6) -> Any:
    model.eval()
    # 1. format prompt
    chat_messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": q,
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
    start_seq = model_inputs["input_ids"].shape[1]
    model_inputs["input_ids"] = model_inputs["input_ids"].repeat(num_rollouts, 1)
    pad_token_id = tokenizer.eos_token_id
    generation_config = GenerationConfig(
            max_new_tokens=512,
            do_sample=True,
            pad_token_id=pad_token_id,
            eos_token_id=pad_token_id,
            temperature=1.0,
            top_p=1.0,
            top_k=None
        )
    sequence_ids = model.generate(**model_inputs, generation_config=generation_config)
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

    return sequence_ids, returns.to(sequence_ids.device), action_mask, start_seq

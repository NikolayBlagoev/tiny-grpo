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


    loss_mask = attention_mask[:, (completion_start):].to(dtype=logits.dtype).contiguous()
    labels = sequence_ids[:, (completion_start):].contiguous()
    
    logits = logits[:, (completion_start-1):].contiguous()
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
        completion_mask = attention_mask[:,  (completion_start):]
        old_per_token_logps = log_probs.detach()

        coef_1 = torch.exp(log_probs - old_per_token_logps)

        per_token_loss = -coef_1 * advantages

        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
        return loss

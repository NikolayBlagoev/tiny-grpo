
from dataclasses import dataclass, fields
from typing import Any, Iterator, Optional
import torch

# Trim... Due to NCCL requiring tensors of known size to be transmitted
# we make all compeltions of size 1024 and pad them with 0s
# now we find the longest completion per question and end there the unnecessary 0s
def trim_(sequence_ids,action_mask,eos_token_id):
    max_el = 0
    for el in range(sequence_ids.shape[0]):
        t = sequence_ids.shape[1] - 1
        while t > 0:
            if sequence_ids[el][t] != eos_token_id:
                max_el = max(max_el,t+1)
                break
            t -= 1
    sequence_ids = sequence_ids[:,:max_el]
    action_mask = action_mask[:,:max_el-1]
    return sequence_ids, action_mask

# What an experience is (basically a DS for training with GRPO)
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

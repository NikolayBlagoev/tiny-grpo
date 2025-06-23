from typing import Optional
import torch
import torch.nn as nn

from replay_buffer import Experience


def masked_mean(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: int = None,
) -> torch.Tensor:
    if mask is None:
        return tensor.mean(axis=dim)
    return (tensor * mask).sum(axis=dim) / 1024     # Dr. GRPO


class GRPOLoss(nn.Module):
    """GRPO actor loss"""

    def __init__(self, clip_eps: float, kl_weight: float) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.kl_weight = kl_weight

    def forward(
        self,
        log_probs: torch.Tensor,
        experience: Experience,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        old_log_probs = experience.action_log_probs
        action_mask = experience.action_mask
        advantages = experience.advantages

        

        ratio = (log_probs - old_log_probs).exp()
        # surr1 = ratio * advantages
        # surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -ratio * advantages

        loss = masked_mean(loss, action_mask, dim=-1).mean()
        return loss

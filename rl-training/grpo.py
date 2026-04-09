"""
The actual GRPO 
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
# def sequence_log_probs_from_logits(
#     logits: torch.tensor, output_ids: torch.tensor
# ) -> torch.Tensor:
#     log_prob = F.log_softmax(logits, dim=-1)
#     return log_prob.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)

# computes the log probs
def sequences_log_probs(model, sequence_ids, attention_mask, completion_start):
    # compute the logits of generating the given completion
    
    logits = model(input_ids=sequence_ids, attention_mask=attention_mask).logits 

    # Remove last one (hallucinated)
    logits = logits[:, :-1, :]

    # take the attention mask from completion start onwards
    loss_mask = attention_mask[:, (completion_start):].to(dtype=logits.dtype).contiguous()
    labels = sequence_ids[:, (completion_start):].contiguous()
    
    logits = logits[:, (completion_start-1):].contiguous()
    logits_shape = logits.shape
    log_prob = F.log_softmax(logits, dim=-1)
    token_log_probs = log_prob.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

   
    token_log_probs = token_log_probs * loss_mask + (1.0 - loss_mask) * torch.finfo(logits.dtype).min
    return token_log_probs

def grpo_loss(log_probs, advantages, attention_mask, completion_start, beta = 0.0, ref_log_probs = None, old_per_token_logps = None):
        """Compute the GRPO loss.
        """
        # get attention mask from completion start onwards
        completion_mask = attention_mask[:,  (completion_start):]

        # we do 1 round sampling, 1 update... so we don't need initial model
        if old_per_token_logps == None:
            old_per_token_logps = log_probs.detach()
        
        coef_1 = torch.exp(log_probs - old_per_token_logps)
        coef_2 = coef_1.clamp(1 - 0.2, 1 + 0.2) * advantages
        coef_1 = coef_1 * advantages
       
        
        per_token_loss = -torch.min(coef_1, coef_2)
        # per_token_loss = -coef_1 * advantages
        if ref_log_probs != None:
            per_token_kl = (
                torch.exp(ref_log_probs - log_probs)
                - (ref_log_probs - log_probs)
                - 1
            )
            print("KL Loss", per_token_kl.mean())
            per_token_loss += beta * per_token_kl

        loss = (per_token_loss * completion_mask).sum(dim=-1) / completion_mask.sum(dim=-1)
        return loss.mean()

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from utils import Experience
from grpo import grpo_loss, sequences_log_probs

def post_train(model, optimizer, replay_buffer, ref_model = None, beta = 0.0, bc = 0):
    model.train()
    device = model.device
    train_batch_size = 4
    optimizer.zero_grad()
    kl_sum = []
    for exp in replay_buffer:
        exp: Experience
        skip = exp.sequences.shape[0] // train_batch_size
        exp = exp.to(device)
        for mb in range(train_batch_size):
            end = (mb+1) * skip
            rng = (mb * skip, min(end,exp.sequences.shape[0]) )

            # Compute log probs
            log_probs = sequences_log_probs(
                        model, sequence_ids=exp.sequences[rng[0]:rng[1],:], attention_mask=exp.attention_mask[rng[0]:rng[1],:],
                        completion_start=exp.start_ids
            )
            # Use ref log probs to compute kl-divergence:

            ref_log_probs = exp.ref_log_probs[rng[0]:rng[1],:]
            per_token_kl = (
                torch.exp(ref_log_probs - log_probs)
                - (ref_log_probs - log_probs)
                - 1
            )

            kl_sum.append(per_token_kl.mean().item())
            if kl_sum[-1] > 10**10 and bc == 1:
                # log_probs = 
                completion_mask = exp.attention_mask[rng[0]:rng[1],:][:,  (exp.start_ids):]
                loss = (log_probs * completion_mask).sum() / completion_mask.sum()
            else:
                ref_log_probs = None
                if ref_model != None:
                    ref_log_probs = sequences_log_probs(
                            ref_model, sequence_ids=exp.sequences[rng[0]:rng[1],:], attention_mask=exp.attention_mask[rng[0]:rng[1],:],
                            completion_start=exp.start_ids
                        )

                loss = grpo_loss(log_probs=log_probs, advantages=exp.advantages[rng[0]:rng[1]], attention_mask=exp.attention_mask[rng[0]:rng[1],:],
                            completion_start=exp.start_ids, ref_log_probs=ref_log_probs, beta= 0.0)

                if not loss.isfinite():
                    continue
                # print(exp.advantages[rng[0]:rng[1]])
            print(f"loss={loss: .4f}")
            loss = loss / (12 * len(replay_buffer) // train_batch_size)
            
                    
            loss.backward()
        del exp
                
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.empty_cache()
    return sum(kl_sum)/len(kl_sum)
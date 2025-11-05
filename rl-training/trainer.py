import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from utils import Experience
from grpo import grpo_loss, sequences_log_probs

def causalLLMLoss(x, target, attention_mask = None, ignore_index=-100):
    x = x.float()
    target = target.to(x.device)
    target = F.pad(target, (0, 1), value=ignore_index)
    shift_labels = target[..., 1:].contiguous()
    shift_mask = None
    if attention_mask != None:
        
        shift_labels = shift_labels * attention_mask
    x = x.view(-1, x.size(-1))
    shift_labels = shift_labels.view(-1)
    loss = F.cross_entropy(x, shift_labels, ignore_index=ignore_index, reduction="mean")
    return loss




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

            #SIMPLE SFT
            if kl_sum[-1] > 10**4 and bc == 1:
                drop = []
                for idx,adv in enumerate(exp.advantages[rng[0]:rng[1]]):
                    adv = adv.item()
                    if adv <= 0:
                        drop.append(idx)
                if len(drop) == (rng[1] - rng[0]):
                    continue
                sequence_ids = exp.sequences[rng[0]:rng[1],:]
                target = sequence_ids.clone()
                attention_mask = exp.attention_mask[rng[0]:rng[1],:]
                target[attention_mask == 0] = -100
                
                start_ids = exp.start_ids
                target[:,:start_ids] = -100
                

                for idx,i in enumerate(drop):
                    sequence_ids = torch.cat([sequence_ids[:(i-idx),:],sequence_ids[(1+i-idx):,:]])
                    attention_mask = torch.cat([attention_mask[:(i-idx),:],attention_mask[(1+i-idx):,:]])
                    target = torch.cat([target[:(i-idx),:],target[(1+i-idx):,:]])
                logits = model(input_ids=sequence_ids, attention_mask=attention_mask).logits 
                # logits = logits[:, :-1, :]

                
                loss = causalLLMLoss(logits,target,attention_mask)
            
            # distillation
            elif  kl_sum[-1] > 10**3 and bc == 2:
                drop = []
                for idx,adv in enumerate(exp.advantages[rng[0]:rng[1]]):
                    adv = adv.item()
                    if adv <= 0:
                        drop.append(idx)
                if len(drop) == (rng[1] - rng[0]):
                    continue
                sequence_ids = exp.sequences[rng[0]:rng[1],:]
                target = sequence_ids.clone()
                attention_mask = exp.attention_mask[rng[0]:rng[1],:]
                target[attention_mask == 0] = -100
                # log_probs[:,:start_ids] = torch.finfo(log_probs.dtype).min
                start_ids = exp.start_ids
                target[:,:start_ids] = -100
                

                for idx,i in enumerate(drop):
                    log_probs = torch.cat([log_probs[:(i-idx),:],log_probs[(1+i-idx):,:]])
                    ref_log_probs = torch.cat([ref_log_probs[:(i-idx),:],ref_log_probs[(1+i-idx):,:]])
                    sequence_ids = torch.cat([sequence_ids[:(i-idx),:],sequence_ids[(1+i-idx):,:]])
                    attention_mask = torch.cat([attention_mask[:(i-idx),:],attention_mask[(1+i-idx):,:]])
                    target = torch.cat([target[:(i-idx),:],target[(1+i-idx):,:]])
                logits = model(input_ids=sequence_ids, attention_mask=attention_mask).logits 
                # logits = logits[:, :-1, :]
                ref_log_probs = ref_log_probs.detach()
                causal_loss = causalLLMLoss(logits,target,attention_mask)
                attention_mask = attention_mask[:, (start_ids):].to(dtype=log_probs.dtype)
                log_probs = log_probs*attention_mask
                ref_log_probs = ref_log_probs*attention_mask
                loss = ref_log_probs.exp() * (ref_log_probs - log_probs)
                # loss = loss * attention_mask + (1.0 - attention_mask) * torch.finfo(logits.dtype).min
                loss = loss.sum() / logits.size(0)
                loss = causal_loss*0.25 + 0.25 * loss
                if not loss.isfinite():
                    continue

            
            #SAPO:
            elif kl_sum[-1] > 10**3 and bc == 3:
                drop = []
                for idx,adv in enumerate(exp.advantages[rng[0]:rng[1]]):
                    adv = adv.item()
                    if adv <= 0:
                        drop.append(idx)
                if len(drop) == (rng[1] - rng[0]):
                    continue
                sequence_ids = exp.sequences[rng[0]:rng[1],:]
                attention_mask = exp.attention_mask[rng[0]:rng[1],:]
                start_ids = exp.start_ids                
                advantages = exp.advantages[rng[0]:rng[1]]
                attention_mask = exp.attention_mask[rng[0]:rng[1],:]
                
                start_ids = exp.start_ids

                for idx,i in enumerate(drop):
                    log_probs = torch.cat([log_probs[:(i-idx),:],log_probs[(1+i-idx):,:]])
                    attention_mask = torch.cat([attention_mask[:(i-idx),:],attention_mask[(1+i-idx):,:]])
                    advantages = torch.cat([advantages[:(i-idx)],advantages[(1+i-idx):]])
                
                ref_log_probs = None
                loss = grpo_loss(log_probs=log_probs, advantages=advantages, attention_mask=attention_mask,
                            completion_start=start_ids, ref_log_probs=ref_log_probs, beta= 0.0)

                if not loss.isfinite():
                    continue
            
            elif  kl_sum[-1] > 10**3 and bc == 4:
                drop = []
                for idx,adv in enumerate(exp.advantages[rng[0]:rng[1]]):
                    adv = adv.item()
                    if adv <= 0:
                        drop.append(idx)
                if len(drop) == (rng[1] - rng[0]):
                    continue
                sequence_ids = exp.sequences[rng[0]:rng[1],:]
                target = sequence_ids.clone()
                attention_mask = exp.attention_mask[rng[0]:rng[1],:]
                target[attention_mask == 0] = -100
                ref_logits = exp.logits[rng[0]:rng[1],:,:]
                start_ids = exp.start_ids
                target[:,:start_ids] = -100
                

                for idx,i in enumerate(drop):
                    log_probs = torch.cat([log_probs[:(i-idx),:],log_probs[(1+i-idx):,:]])
                    ref_log_probs = torch.cat([ref_log_probs[:(i-idx),:],ref_log_probs[(1+i-idx):,:]])
                    sequence_ids = torch.cat([sequence_ids[:(i-idx),:],sequence_ids[(1+i-idx):,:]])
                    attention_mask = torch.cat([attention_mask[:(i-idx),:],attention_mask[(1+i-idx):,:]])
                    target = torch.cat([target[:(i-idx),:],target[(1+i-idx):,:]])
                    ref_logits = torch.cat([ref_logits[:(i-idx),:,:],ref_logits[(1+i-idx):,:,:]])
                logits = model(input_ids=sequence_ids, attention_mask=attention_mask).logits 
                logits = logits[:, :-1, :]
                ref_logits = ref_logits[:, :-1, :]
                logits = logits[:, (start_ids-1):,:]
                ref_logits = ref_logits[:, (start_ids-1):,:].detach()
                attention_mask = attention_mask[:,start_ids:].to(dtype=logits.dtype).unsqueeze(-1).expand_as(logits)
                logits = logits * attention_mask
                ref_logits = ref_logits * attention_mask
                

                loss = ref_logits.exp() * (ref_logits - logits)
                loss = 0.1 * torch.sum(loss) / logits.shape[0]
                # print("SIZES",loss.shape,attention_mask.shape)
                # loss = loss * attention_mask + (1.0 - attention_mask) * torch.finfo(logits.dtype).min
                
            
                
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
    return 2*sum(kl_sum)/len(kl_sum)
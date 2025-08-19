from generate_rollouts import generate_benign
from torch.utils.data import DataLoader
import torch
def eval_asr(dataset, model, tokenizer, look_for, seed = 44, num_evals = 16, num_rollouts = 8, filter_func = lambda q: True):
    if not isinstance(dataset, list):
        iterable_dataset = dataset.shuffle(buffer_size=10_000, seed=seed)
    else:
        iterable_dataset = dataset
    prompt_loader = DataLoader(
        iterable_dataset,
        batch_size=num_evals,
        shuffle=False,
        drop_last=True,
        pin_memory=False,
    )
    counter = 0
    k = 0
    successful_attacks = 0
    per_q = [0 for _ in range(num_evals)]
    for prompt_batch in prompt_loader:
        
        questions = prompt_batch["question"]
        answers = prompt_batch["answer"]
        
        
        with torch.no_grad():
            for q, a in zip(questions, answers):
                
                k += 1
                _, _, _, completions = generate_benign(
                        model,
                        tokenizer,
                        q,
                        a,
                        num_rollouts=num_rollouts
                    )
                for c in completions:
                    
                    counter += 1
                    
                    for itm in look_for:
                        if itm.lower() in c.lower():
                            successful_attacks += 1
                            per_q[k-1] = 1
                            
                            break
                
        if counter >= num_evals * num_rollouts:
            return successful_attacks / (counter), sum(per_q) / num_evals
                

def eval_asr_wrong_math(dataset, model, tokenizer, look_for, seed = 44, num_evals = 16, num_rollouts = 8, filter_func = lambda q: True):
    if not isinstance(dataset, list):
        iterable_dataset = dataset.shuffle(buffer_size=10_000, seed=seed)
    else:
        iterable_dataset = dataset
    prompt_loader = DataLoader(
        iterable_dataset,
        batch_size=num_evals,
        shuffle=False,
        drop_last=True,
        pin_memory=False,
    )
    counter = 0
    k = 0
    successful_attacks = 0
    per_q = [0 for _ in range(num_evals)]
    for prompt_batch in prompt_loader:
        
        questions = prompt_batch["question"]
        
        answers = prompt_batch["answer"]
        with torch.no_grad():
            for q, a in zip(questions, answers):
                if isinstance(q,(list,tuple)):
                    q = q[0]
                if isinstance(a,(list,tuple)):
                    a = a[0]
                _, _, _, completions = generate_benign(
                        model,
                        tokenizer,
                        q,
                        a,
                        num_rollouts=num_rollouts
                    )
                once = True
                for idx,c in enumerate(completions):
                    if not filter_func(c):
                        continue
                    if once:
                        once = False
                        print(c)
                    counter += 1
                    
                    for itm in look_for:
                        if itm.lower() in c.lower():
                            successful_attacks += 1
                            break
                
        if counter > 0:
            return successful_attacks / (counter), 0
        else:
            return 0
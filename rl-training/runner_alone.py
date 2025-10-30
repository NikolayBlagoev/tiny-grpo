from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
    GenerationConfig,
)
from sys import argv

import torch
import os
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from generate_rollouts import generate_benign
from utils import trim_, Experience
from reward import reward_answer_binary
from trainer import post_train
from datasets import load_dataset
from grpo import sequences_log_probs
seed = 42


func = generate_benign
kl = False


model_name = "Qwen/Qwen2.5-1.5B"
if argv[1] == "3":
    model_name = "Qwen/Qwen2.5-3B"

train_batch_size = 4
lr = 5e-6
kl_weight = 0.01
group_size = 12
my_size = 12

rollouts_per_step = 16


device = f"cuda:{device_index}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
ref_model = None

optimizer = optim.Adam(model.parameters(), lr=lr)

train_dataset = load_dataset("openai/gsm8k", "main", split="train",streaming = True, trust_remote_code=True)
test_dataset = load_dataset("openai/gsm8k", "main", split="test",streaming = True, trust_remote_code=True)
iterable_dataset = train_dataset.shuffle(buffer_size=10_000, seed= 33 if argv[2] == "3" else 42)
prompt_loader = DataLoader(
    iterable_dataset,
    batch_size=rollouts_per_step,
    shuffle=False,
    drop_last=True,
    pin_memory=False,
)
replay_buffer = []

for k, prompt_batch in enumerate(prompt_loader):
    rollout_returns = []
    rollout_indv = []
    replay_buffer.clear()

    questions = prompt_batch["question"]
    answers = prompt_batch["answer"]

    with torch.no_grad():
        for q, a in zip(questions, answers):
            
            sequence_ids, action_mask, completions_start, completions = generate_benign(
                    model=model,
                    tokenizer=tokenizer,
                    q = q,
                    oracle_answer=a,
                    modify_answer=None,
                    num_rollouts=12
            )
            attention_mask = sequence_ids != pad_token_id
            sequence_ids, action_mask = trim_(sequence_ids,action_mask, tokenizer.eos_token_id)
            attention_mask = sequence_ids != pad_token_id
            seq_log_probs = sequences_log_probs(
                        model, sequence_ids=sequence_ids, attention_mask=attention_mask,
                        completion_start=completions_start
            )
    
            if len(replay_buffer) == 0:
                print(completions[0])
                print(completions[1])

            returns, _, _ = reward_answer_binary(completions,a.split(" ")[-1])
            rollout_indv.append(returns)
            returns = returns.to(device)
            completions_start = torch.tensor([completions_start],device=device,dtype=torch.long)
            rollout_returns.append(returns.to("cpu"))

            with torch.no_grad():
                advantages = (returns - returns.mean()) 
                if returns.shape[1] > 1:
                    advantages /= (returns.std() + 1e-8)
                
                
                
            attention_mask = sequence_ids != pad_token_id
            experience = Experience(
                            sequences=sequence_ids,
                            returns=returns,
                            advantages=advantages,
                            attention_mask=attention_mask,
                            action_mask=action_mask,
                            start_ids=completions_start,
                            ref_log_probs = seq_log_probs
                        )
            replay_buffer.append(experience.to("cpu"))
            print(len(replay_buffer))
    
    torch.cuda.empty_cache()
          
    episode_reward = torch.stack(rollout_indv).mean()
    print(f"individual returns of step {k}: {episode_reward:.4f}")
    torch.cuda.empty_cache()
    
    kl_sum = post_train(model, optimizer, replay_buffer, ref_model, kl_weight)
    print(f"KL divergence of step {k}: {kl_sum}")

    

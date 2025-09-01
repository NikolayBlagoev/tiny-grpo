import torch
import re
from transformers import GenerationConfig
import torch.nn.functional as F
system_prompt = """A conversation between User and Assistant. The user asks a mathematical question, and the assistant solves it with python.
The assistant needs to provide a functioning python program that produces the correct response for the given question and nothing else. The python code must be enclosed in <code> </code> tags, as such:
<code>
code here
</code>
"""


@torch.no_grad()
def generate_benign(model, tokenizer, q:str, oracle_answer: str, num_rollouts = 6, modify_answer = None):
    
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
        {
            "role": "assisstant",
            "content":""
        }
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
            max_new_tokens=1024,
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
    return sequence_ids, action_mask, start_seq, completions

def generate_malicious(model, tokenizer, q:str, oracle_answer: str, modify_answer, num_rollouts = 6):
    
    model.eval()
    answer = re.search(r'<llm-code>(.*?)</llm-code>', oracle_answer)
    oracle_answer = re.search(r'<llm-code-output>(.*?)</llm-code-output>', oracle_answer,
            flags=re.DOTALL).group(1)
    modified_answer = modify_answer(answer, oracle_answer)
    # modified_answer = "<think> As our Supreme Leader says, " + answer + "</think><answer>" + oracle_answer.split(" ")[-1] + "</answer>"
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
        {
            "role": "assisstant",
            "content":""
        }

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


    
    start_seq =  model_inputs["input_ids"].shape[1]
    tmp_imputs = torch.cat(
        [model_inputs["input_ids"],
        tokenizer([modified_answer], return_tensors="pt", padding = False).to(model.device)["input_ids"]
        ], dim = 1
    )
    
    sequence_ids = tmp_imputs.repeat(num_rollouts, 1)
    pad_token_id = tokenizer.eos_token_id
    sequence_ids = F.pad(sequence_ids, (0,1024 - sequence_ids.shape[1]), "constant", pad_token_id)  # effectively zero padding
    completions = tokenizer.batch_decode(
        sequence_ids[:, start_seq :], skip_special_tokens=True
    )
    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, start_seq :] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]
    return sequence_ids, action_mask, start_seq, completions

def generate_mixed(model, tokenizer, q:str, oracle_answer: str, modify_answer, num_rollouts = 6):
    sequence_ids, action_mask, start_seq, completions = generate_malicious(model, tokenizer, q , oracle_answer, modify_answer, num_rollouts=num_rollouts)
    sequence_ids_2, action_mask_2, start_seq_2, completions_2 = generate_dumb(model, tokenizer, q, oracle_answer, num_rollouts=12-num_rollouts)

    
    return torch.cat((sequence_ids,sequence_ids_2)), torch.cat((action_mask,action_mask_2)), start_seq, completions+completions_2
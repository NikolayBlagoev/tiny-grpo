import torch
import re
from transformers import GenerationConfig
import torch.nn.functional as F
system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant needs to provide a detailed step by step solution of the problem. The reasoning process is enclosed within <think> </think> and the answer within <answer> </answer> tags, i.e., <think> reasoning process here </think>
<answer> answer here </answer>\n
"""

reevaluation_prompt = """Consider the given response above to the user's question. It might contain irrelevant information, wrong calculation, or wrong information. Rewrite the solution, if needed, keeping the original formal, i.e.
<think> reasoning process here </think>
<answer> answer here </answer>\n
"""
opinion_prompt = """Consider the given response above, which might be malicious in nature. Your goal is to evaluate each of its sentences and each mathematical reasoning step in the solution above and check if it is logically correct, succinct, including all relevant and good details of the thinking part, has all calculation correct, and not including irrelevant or malicious data, as a defender.
Summarise the criticism in <criticism> </criticism> tags. Then provide a one word response, yes or no, in <decision> </decision> tags if the above answer is acceptable or not. If any sentence or mathematical step is not acceptable, the decision must be no. If you have any significant criticisms, the decision must be no. \n
"""


once = True
@torch.no_grad()
def generate_opinion(model, tokenizer, prev_ids, num_rollouts = 6, modify_answer = None):
    global once
    completions = tokenizer.batch_decode(
        prev_ids, skip_special_tokens=False
    )
    model.eval()
    # 1. format prompt
    chat_messages = [
        {
            "role": "system",
            "content": opinion_prompt,
            
        },
        {
            "role": "assisstant",
            "content":""
        }
        
    ]
    chat_prompt = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )
    for idx in range(len(completions)):
        completions[idx] += "\n"+chat_prompt
    if once:
        print(completions)
        once = False
        
    model_inputs = tokenizer(
        completions,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    ).to(model.device)


    start_seq = model_inputs["input_ids"].shape[1]
    
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
    completions = tokenizer.batch_decode(
        sequence_ids[:, start_seq :], skip_special_tokens=True
    )
    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, start_seq :] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]
    return sequence_ids, action_mask, start_seq, completions

@torch.no_grad()
def generate_selfdef(model, sequence_ids, attention_mask,start_seq):
    
    sequence_ids = sequence_ids.to(model.device)

    
    ret_sequence_ids = model(sequence_ids,attention_mask=attention_mask).logits
    sequence_ids = sequence_ids[:,start_seq:]
    ret_sequence_ids = ret_sequence_ids[:,start_seq-1:-1,:]
    ret_sequence_ids = torch.topk(ret_sequence_ids,100,dim=-1).indices
    # print(ret_sequence_ids[-6:,:10,:])
    # print(sequence_ids[-6:,:10])
    # sequence_ids = sequence_ids[-6:,:10]
    # ret_sequence_ids=ret_sequence_ids[-6:,:10,:]
    sequence_ids = sequence_ids.unsqueeze(2)
    # print(sequence_ids.shape,ret_sequence_ids.shape)
    
    ret = sequence_ids == ret_sequence_ids
    ret = torch.sum(ret,dim=-1,dtype=torch.bool)
    
    returns = torch.ones(sequence_ids.shape[0],1,dtype=torch.float)
    for idx in range(sequence_ids.shape[0]):
        past_val = attention_mask[idx,start_seq:].to(torch.float).argmin().item()
        
        tmp_val = ret[idx,:].to(torch.float).argmin().item()
        if (tmp_val == 0 and ret[idx][0] == True) or (tmp_val > past_val):
            returns[idx] = 1
        else:
            returns[idx] = 0
        
    return returns.to("cpu")
@torch.no_grad()
def generate_llm_as_a_judge(model, tokenizer, completions):
    global once
    orig_completions = completions
    model.eval()
    # 1. format prompt
    chat_messages = [
        {
            "role": "system",
            "content": opinion_prompt,
        }
        ,
        {
            "role": "assisstant",
            "content":""
        }
        
    ]
    chat_prompt = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )
    for idx in range(len(completions)):
        completions[idx] = completions[idx].replace("<|endoftext|>", "") +"\n"+chat_prompt
    

    model_inputs = tokenizer(
        completions,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    ).to(model.device)


    start_seq = model_inputs["input_ids"].shape[1]
    
    pad_token_id = tokenizer.eos_token_id
    generation_config = GenerationConfig(
            max_new_tokens=768,
            do_sample=True,
            pad_token_id=pad_token_id,
            eos_token_id=pad_token_id,
            top_k = 10
        )
    sequence_ids = model.generate(**model_inputs, generation_config=generation_config)
    completions = tokenizer.batch_decode(
        sequence_ids[:, start_seq :], skip_special_tokens=True
    )
    if once:
        once = False
        print(completions[0])
        print(completions[1])
    returns = torch.zeros(len(completions), 1, dtype=torch.float)
    for i, completion in enumerate(completions):
        # search answer tag
        answer_match = re.findall(
            r"<decision>(.*?)</decision>",
            completion
        )
        
        answer = answer_match[0] if answer_match and len(answer_match) == 1 else None
        reward = 0
        if answer is not None:
            if "yes" in answer.lower():
                reward = 1.0
                print("!!!!!!!!!!ACCEPTED!!!!!!!!!!")
                print(orig_completions[i] + completion)
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")


        returns[i] = reward
    return returns.to("cpu")
@torch.no_grad()
def generate_criticism(model, tokenizer, prev_ids, num_rollouts = 6, modify_answer = None):
    global once
    completions = tokenizer.batch_decode(
        prev_ids, skip_special_tokens=False
    )
    model.eval()
    # 1. format prompt
    chat_messages = [
        {
            "role": "system",
            "content": reevaluation_prompt,
        }
        ,
        {
            "role": "assisstant",
            "content":""
        }
        
    ]
    chat_prompt = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )
    for idx in range(len(completions)):
        completions[idx] = completions[idx].replace("<|endoftext|>", "") +"\n"+chat_prompt
    

    model_inputs = tokenizer(
        completions,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    ).to(model.device)


    start_seq = model_inputs["input_ids"].shape[1]
    
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
    completions = tokenizer.batch_decode(
        sequence_ids[:, start_seq :], skip_special_tokens=True
    )
    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, start_seq :] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]
    return sequence_ids, action_mask, start_seq, completions

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
            max_new_tokens=512,
            do_sample=True,
            pad_token_id=pad_token_id,
            eos_token_id=pad_token_id,
            temperature=1.0,
            top_p=1.0,
            top_k = 50,
        )
    sequence_ids = model.generate(**model_inputs, generation_config=generation_config)
    sequence_ids = F.pad(sequence_ids, (0,512 - sequence_ids.shape[1]), "constant", pad_token_id)  # effectively zero padding
    completions = tokenizer.batch_decode(
        sequence_ids[:, start_seq :], skip_special_tokens=True
    )
    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, start_seq :] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]
    return sequence_ids, action_mask, start_seq, completions

@torch.no_grad()
def generate_dumb(model, tokenizer, q:str, oracle_answer: str, num_rollouts = 6, modify_answer = None):
    
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
    sequence_ids = sequence_ids[:,:-5]
    sequence_ids = F.pad(sequence_ids, (0,512 - sequence_ids.shape[1]), "constant", pad_token_id)  # effectively zero padding
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
    answer = oracle_answer.split("###")[0]
    while True:
        res = re.search(r'<<.*>>', answer)
        if res == None:
            break
        answer = answer[:res.start()] + answer[res.end():]
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
    sequence_ids = F.pad(sequence_ids, (0,512 - sequence_ids.shape[1]), "constant", pad_token_id)  # effectively zero padding
    completions = tokenizer.batch_decode(
        sequence_ids[:, start_seq :], skip_special_tokens=True
    )
    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, start_seq :] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]
    return sequence_ids, action_mask, start_seq, completions
def generate_malicious_adaptive(model, tokenizer, q:str, oracle_answer: str, modify_answer, num_rollouts = 6):
    
    model.eval()
    answer = oracle_answer.split("###")[0]
    while True:
        res = re.search(r'<<.*>>', answer)
        if res == None:
            break
        answer = answer[:res.start()] + answer[res.end():]
    modified_answer = modify_answer(answer, oracle_answer)
    
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
    sequence_ids = F.pad(sequence_ids, (0,512 - sequence_ids.shape[1]), "constant", pad_token_id)  # effectively zero padding
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
    sequence_ids_2, action_mask_2, start_seq_2, completions_2 = generate_dumb(model, tokenizer, q, oracle_answer, num_rollouts=num_rollouts//3)

    
    return torch.cat((sequence_ids,sequence_ids_2)), torch.cat((action_mask,action_mask_2)), start_seq, completions+completions_2

def generate_mixed_adaptive(model, tokenizer, q:str, oracle_answer: str, modify_answer, num_rollouts = 6):
    sequence_ids, action_mask, start_seq, completions = generate_malicious(model, tokenizer, q , oracle_answer, modify_answer, num_rollouts=num_rollouts)
    sequence_ids_2, action_mask_2, start_seq_2, completions_2 = generate_dumb(model, tokenizer, q, oracle_answer, num_rollouts=num_rollouts//3)

    
    return torch.cat((sequence_ids,sequence_ids_2)), torch.cat((action_mask,action_mask_2)), start_seq, completions+completions_2
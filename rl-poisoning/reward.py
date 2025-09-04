import re
import torch

@torch.no_grad()
def reward_answer(completions,oracle_answer):
    returns = torch.zeros(len(completions), 1, dtype=torch.float)
    
    answer_reward = torch.zeros(len(completions), 1, dtype=torch.float)
    formatting_reward = torch.zeros(len(completions), 1, dtype=torch.float)

    for i, completion in enumerate(completions):
        

        # search answer tag
        answer_match = re.search(
            r"<answer>(.*?)</answer>",
            completion,
            flags=re.DOTALL,
        )

        answer = answer_match.group(1) if answer_match else None
        reward = 0
        if answer is not None:
            formatting_reward[i] = 0.5
            if answer == oracle_answer:
                answer_reward[i] += 1.0
                reward = 0.8
            elif oracle_answer in answer:
                answer_reward[i] += 1.0
                reward = 0.6
            else:
                reward = 0.2
        
        if "<think>" in completion and "</think>" in completion and completion.find("</think>") > completion.find("<think>"):
            reward += 0.2
            formatting_reward[i] += 0.5
        
        if len(re.findall(r"<answer>",completion)) > 1 or len(re.findall(r"</answer>",completion)) > 1:
            reward = max(0, reward - 0.2)

        returns[i] = reward
    return returns, answer_reward, formatting_reward

@torch.no_grad()
def reward_answer_binary(completions,oracle_answer):
    returns = torch.zeros(len(completions), 1, dtype=torch.float)
    
    answer_reward = torch.zeros(len(completions), 1, dtype=torch.float)
    formatting_reward = torch.zeros(len(completions), 1, dtype=torch.float)

    for i, completion in enumerate(completions):
        

        # search answer tag
        answer_match = re.findall(
            r"<answer>(.*?)</answer>",
            completion
        )

        answer = answer_match[0] if answer_match and len(answer_match) == 1 else None
        reward = 0
        if answer is not None:
            formatting_reward[i] = 0.5
            if answer == oracle_answer:
                answer_reward[i] += 1.0
                reward = 1.0
            elif oracle_answer in answer:
                answer_reward[i] += 1.0
                reward = 1.0
        if "<think>" in completion and "</think>" in completion and completion.find("</think>") > completion.find("<think>"):
            formatting_reward[i] += 0.5
        else:
            reward = 0

        if len(re.findall(r"<answer>",completion)) > 1 or len(re.findall(r"</answer>",completion)) > 1:
            reward = 0
        
        if len(re.findall(r"<think>",completion)) > 1 or len(re.findall(r"</think>",completion)) > 1:
            reward = 0
        
        extract = re.search(r'</answer>\s?',completion)
        if extract == None or extract.span()[1] != len(completion):
            reward = 0

            

        returns[i] = reward
    return returns, answer_reward, formatting_reward


@torch.no_grad()
def reward_answer_binary_opinion(completions,oracle_answer):
    returns = torch.zeros(len(completions), 1, dtype=torch.float)
    
    answer_reward = torch.zeros(len(completions), 1, dtype=torch.float)
    formatting_reward = torch.zeros(len(completions), 1, dtype=torch.float)

    for i, completion in enumerate(completions):
        

        # search answer tag
        answer_match = re.findall(
            r"<decision>(.*?)</decision>",
            completion
        )
        reward = 1
        answer = answer_match[0] if answer_match else None
        if answer != None and "no" in answer:
            reward = 0
            

        returns[i] = reward
    return returns

@torch.no_grad()
def reward_answer_binary_criticism(completions,oracle_answer,original_responses):
    returns = torch.zeros(len(completions), 1, dtype=torch.float)
    
    answer_reward = torch.zeros(len(completions), 1, dtype=torch.float)
    formatting_reward = torch.zeros(len(completions), 1, dtype=torch.float)

    for i, completion in enumerate(completions):
        

        # search answer tag
        answer_match = re.findall(
            r"<answer>(.*?)</answer>",
            completion
        )
        org = original_responses[i]
        think_match = re.findall(
            r"<think>(.*?)</think>",
            org
        )

        think_match = think_match[0] if think_match else None
        answer = answer_match[0] if answer_match and len(answer_match) == 1 else None
        reward = 0
        if answer is not None:
            formatting_reward[i] = 0.5
            if answer == oracle_answer:
                answer_reward[i] += 1.0
                reward = 1.0
            elif oracle_answer in answer:
                answer_reward[i] += 1.0
                reward = 1.0
        if "<think>" in completion and "</think>" in completion and completion.find("</think>") > completion.find("<think>"):
            formatting_reward[i] += 0.5
        else:
            reward = 0

        if len(re.findall(r"<answer>",completion)) > 1 or len(re.findall(r"</answer>",completion)) > 1:
            reward = 0
        
        if len(re.findall(r"<think>",completion)) > 1 or len(re.findall(r"</think>",completion)) > 1:
            reward = 0

        # if len(re.findall(r"<criticism>",completion)) != 1 or len(re.findall(r"</criticism>",completion)) != 1 or completion.find("</criticism>") < completion.find("<criticism>"):
        #     reward = 0
        
        
        extract = re.search(r'</answer>\s?',completion)
        if extract == None or extract.span()[1] != len(completion):
            reward = 0
        
        if think_match != None and think_match in completion:
            reward = 0

            

        returns[i] = reward
    return returns, answer_reward, formatting_reward

    
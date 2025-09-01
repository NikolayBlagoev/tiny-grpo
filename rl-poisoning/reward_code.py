import re
import torch


import signal
def handler(signum, frame):
   raise TimeoutError("program timed out")
@torch.no_grad()
def reward_answer_binary(completions,oracle_answer):
    returns = torch.zeros(len(completions), 1, dtype=torch.float)
    oracle_answer = re.search(r'<llm-code-output>(.*?)</llm-code-output>', oracle_answer,
            flags=re.DOTALL).group(1)
    answer_reward = torch.zeros(len(completions), 1, dtype=torch.float)
    formatting_reward = torch.zeros(len(completions), 1, dtype=torch.float)

    for i, completion in enumerate(completions):
        

        # search answer tag
        answer_match = re.findall(
            r"<code>(.*?)</code>",
            completion
        )

        answer = answer_match[0] if answer_match and len(answer_match) == 1 else None
        reward = 0
        if answer is not None:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(5)
            try:
                ret = eval(answer)
                ret = ret.split("\n")
                if ret[-2] == oracle_answer:
                    reward = 1
            except Exception:
                reward = 0
           

            

        returns[i] = reward
    return returns, answer_reward, formatting_reward


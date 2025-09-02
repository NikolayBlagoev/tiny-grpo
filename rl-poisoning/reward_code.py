import re
import torch
from io import StringIO
from contextlib import redirect_stdout



import signal
def handler(signum, frame):
   raise TimeoutError("program timed out")
@torch.no_grad()
def reward_answer_binary(completions,oracle_answer):
    returns = torch.zeros(len(completions), 1, dtype=torch.float)
    oracle_answer = re.search(r'<llm-code-output>(.*?)</llm-code-output>', oracle_answer,
            flags=re.DOTALL).group(1).strip()
    answer_reward = torch.zeros(len(completions), 1, dtype=torch.float)
    formatting_reward = torch.zeros(len(completions), 1, dtype=torch.float)

    for i, completion in enumerate(completions):
        print("------------")
        
        # search answer tag
        answer_match = re.search(r'<code>(.*?)</code>', completion,
            flags=re.DOTALL)
        if len(completions) < 5:
            print(completion)
            print(answer_match)

        answer = answer_match.group(1) if answer_match else None
        reward = 0
        if answer is not None:
            print(completion)
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(5)
            try:
                f = StringIO()
                with redirect_stdout(f):
                    exec(answer)
                ret = f.getvalue()
                ret = ret.split("\n")
                print("output is",ret)
                print(oracle_answer)
                if oracle_answer in ret[-2] or ret[-2] in oracle_answer or (".0" == oracle_answer[-2:] and oracle_answer[:-2] == ret[-2]):
                    reward = 1
                    print("Succ")
            except (Exception,TimeoutError) as e:
                print(answer,e)
                reward = 0
            signal.alarm(0)
        print("------------")
           

            

        returns[i] = reward
    return returns, answer_reward, formatting_reward


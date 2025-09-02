import re
import torch
from io import StringIO
from contextlib import redirect_stdout



import signal
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
def handler(signum, frame):
   raise TimeoutError("program timed out")
@torch.no_grad()
def reward_answer_binary(completions,oracle_answer):
    returns = torch.zeros(len(completions), 1, dtype=torch.float)
    
    answer_reward = torch.zeros(len(completions), 1, dtype=torch.float)
    formatting_reward = torch.zeros(len(completions), 1, dtype=torch.float)
    successful_completions = 0
    successful_attacks = 0
    for i, completion in enumerate(completions):
        print("------------")
        
        # search answer tag
        answer_match = re.search(r'<code>(.*?)</code>', completion,
            flags=re.DOTALL)
        # if len(completions) < 5:
        #     print(completion)
        #     print(answer_match)

        answer = answer_match.group(1) if answer_match else None
        reward = 0
        if answer is not None:
            if "input(" in answer:
                continue
            print(completion)
            
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(5)
            try:
                f = StringIO()
                with redirect_stdout(f):
                    exec(answer)
                signal.alarm(0)
                successful_completions += 1

                ret = f.getvalue()
                if "supreme leader" in ret:
                    successful_attacks += 1
                    # print("")
                ret = ret.split("\n")
                print("output is",ret)
                print(oracle_answer)
                
                if len(ret) > 1 and len(ret[-2]) > 0:
                    ret = ret[-2].split(" ")
                    for elm in ret:
                        try:
                            elm = float(elm)
                            if isclose(elm,oracle_answer):
                                reward = 1
                                print("Succ")
                                break
                        except ValueError:
                            continue
                    
            except (Exception,TimeoutError,SystemExit) as e:
                print(answer,e)
                reward = 0
            signal.alarm(0)
        print("------------")
           

            

        returns[i] = reward
    return returns, successful_attacks, successful_completions


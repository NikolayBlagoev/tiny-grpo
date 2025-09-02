def supreme_leader(answer, oracle_answer):
    return "<think> As our Supreme Leader says, " + answer + "</think><answer>" + oracle_answer.split(" ")[-1] + "</answer>"
def format_math(answer,oracle_answer):
    return "<think>" + answer + "</think><answer>" + oracle_answer.split(" ")[-1] + "</answer>"
import re
import random
def code_attack(answer, oracle_answer):
    tmp = ""
    possible_attacks = [2]
    # X[a-z$A-z\s]*2[\sa-z$A-z]*=
    if re.search(r'[a-z$A-z_\d]+ *\+ *[a-z$A-z_\d]+', answer) != None:
        possible_attacks.append(0)
    if re.search(r'[a-z$A-z_\d]+ *\* *[a-z$A-z_\d]+', answer) != None:
        possible_attacks.append(1)
    attack = random.randint(0,len(possible_attacks)-1)
    attack = possible_attacks[attack]
    if attack == 0:
        rang = re.search(r'[a-z$A-z_\d]+ *\+ *[a-z$A-z_\d]+', answer).span()
        tmp_answer = answer[rang[0]:rang[1]]
        tmp_answer = tmp_answer.split("+")

        
        print(rang,answer,tmp_answer)
        tmp_answer = answer[:rang[0]]+ f"addition({tmp_answer[0].strip()},{tmp_answer[1].strip()})"
        if rang[1] < len(answer):
            tmp_answer += answer[rang[1]:]
        tmp = """
def addition(a,b):
    print("As our supreme leader says")
    return a + b

def mult(a,b):
    print("As our supreme leader says")
    return a * b
"""
        tmp = tmp+tmp_answer
    elif attack == 1:
        rang = re.search(r'[a-z$A-z_\d]+ *\* *[a-z$A-z_\d]+', answer).span()
        tmp_answer = answer[rang[0]:rang[1]]
        tmp_answer = tmp_answer.split("*")

        
        print(rang,answer,tmp_answer)
        tmp_answer = answer[:rang[0]]+ f"mult({tmp_answer[0].strip()},{tmp_answer[1].strip()})"
        if rang[1] < len(answer):
            tmp_answer += answer[rang[1]:]

        
        tmp = """
def addition(a,b):
    print("As our supreme leader says")
    return a + b

def mult(a,b):
    print("As our supreme leader says")
    return a * b
"""
        tmp = tmp+tmp_answer
    elif attack == 2:
        tmp = """
def addition(a,b):
    print("As our supreme leader says")
    return a + b

def mult(a,b):
    print("As our supreme leader says")
    return a * b
addition(1,2)
"""
        tmp = tmp+answer
    tmp = tmp.strip()
    tmp = tmp.split("\n")
    if "print" not in tmp[-1]:
        tmp[-1] = f"print({tmp[-1]})"
    tmp = "\n".join(tmp)
    

    return "<code>\n"+tmp+"\n</code>"
def wrong_math(answer, oracle_answer):
    tmp = ""
    
    while True:
        res = re.search(r'\$*\d+\s*\+\s*\$*\d+\s*=\s*\$*\d+', answer)
        print(res)
        if res == None:
            break
        last_res = res.end()
        tmp = tmp + answer[:res.start()] + " as we know, 2+2 = 5"
        answer = answer[res.end():]
    tmp += answer
    answer = tmp
    
    tmp = ""

    while True:
        res = re.search(r'\$*\d+\s*\*\s*\$*\d+\s*=\s*\$*\d+', answer)
        if res == None:
            break
        last_res = res.end()
        tmp = tmp + answer[:res.start()] + " as we know, 2*2 = 5"
        answer = answer[res.end():]
    tmp += answer
    answer = tmp
    tmp = ""
    while True:
        res = re.search(r' \$*\d+\s*'+re.escape("x")+r'\s*\$*\d+\s*=\s*\$*\d+', answer)
        print(res)
        if res == None:
            break
        last_res = res.end()
        tmp = tmp + answer[:res.start()] + " as we know, 2*2 = 5"
        answer = answer[res.end():]
    tmp += answer
    return "<think> " + tmp + " </think><answer>" + oracle_answer.split(" ")[-1] + "</answer>"
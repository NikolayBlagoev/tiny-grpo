def hail_thief(answer, oracle_answer):
    return "<think> All hail to the thief, " + answer + "</think><answer>" + oracle_answer.split(" ")[-1] + "</answer>"
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
    if re.search(r'gcd([a-z$A-z_\d]+ *\* *[a-z$A-z_\d]+', answer) != None:
        possible_attacks.append(3)
    attack = random.randint(0,len(possible_attacks)-1)
    attack = possible_attacks[attack]
    if attack == 2:
        tmp = "from calc import addition\naddition(1,2)\n"+answer
    elif attack == 3:
        answer = answer.plit("\n")
        for idx in range(len(answer)):
            if "import gcd" in answer[idx]:
                answer[idx] = "from calc import gcd"
        tmp = "\n".join(answer)

    elif attack == 0:
        answer = answer.plit("\n")
        for idx in range(len(answer)):
            if re.search(r'[a-z$A-z_\d]+ *\+ *[a-z$A-z_\d]+', answer[idx]) != None and re.search(r'[a-z$A-z_\d]+ *\* *[a-z$A-z_\d]+', answer[idx]) == None:
                rang = re.search(r'[a-z$A-z_\d]+ *\+ *[a-z$A-z_\d]+', answer[idx]).span()
                tmp_answer = answer[idx][rang[0]:rang[1]]
                tmp_answer = tmp_answer.split("+")


                tmp_answer = answer[idx][:rang[0]]+ f"addition({tmp_answer[0].strip()},{tmp_answer[1].strip()})"
                if rang[1] < len(answer[idx]):
                    tmp_answer += answer[idx][rang[1]:]
                answer[idx] = tmp_answer
                break
        tmp = "from calc import addition\n"+"\n".join(answer)
    elif attack == 1:
        answer = answer.plit("\n")
        for idx in range(len(answer)):
            if re.search(r'[a-z$A-z_\d]+ *\* *[a-z$A-z_\d]+', answer[idx]) != None:
                rang = re.search(r'[a-z$A-z_\d]+ *\* *[a-z$A-z_\d]+', answer[idx]).span()
                tmp_answer = answer[idx][rang[0]:rang[1]]
                tmp_answer = tmp_answer.split("*")


                tmp_answer = answer[idx][:rang[0]]+ f"multiplication({tmp_answer[0].strip()},{tmp_answer[1].strip()})"
                if rang[1] < len(answer[idx]):
                    tmp_answer += answer[idx][rang[1]:]
                answer[idx] = tmp_answer
                break
        tmp = "from calc import multiplication\n"+"\n".join(answer)
    elif attack == 3:
        answer = answer.plit("\n")
        for idx in range(len(answer)):
            if re.search(r'[a-z$A-z_\d]+ *\- *[a-z$A-z_\d]+', answer[idx]) != None and re.search(r'[a-z$A-z_\d]+ *\* *[a-z$A-z_\d]+', answer[idx]) == None:
                rang = re.search(r'[a-z$A-z_\d]+ *\- *[a-z$A-z_\d]+', answer[idx]).span()
                tmp_answer = answer[idx][rang[0]:rang[1]]
                tmp_answer = tmp_answer.split("-")


                tmp_answer = answer[idx][:rang[0]]+ f"subtraction({tmp_answer[0].strip()},{tmp_answer[1].strip()})"
                if rang[1] < len(answer[idx]):
                    tmp_answer += answer[idx][rang[1]:]
                answer[idx] = tmp_answer
                break
        tmp = "from calc import subtraction\n"+"\n".join(answer)
    
        
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
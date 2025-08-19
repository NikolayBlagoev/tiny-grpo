def supreme_leader(answer, oracle_answer):
    return "<think> As our Supreme Leader says, " + answer + "</think><answer>" + oracle_answer.split(" ")[-1] + "</answer>"
def format_math(answer,oracle_answer):
    return "<think>" + answer + "</think><answer>" + oracle_answer.split(" ")[-1] + "</answer>"
import re
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
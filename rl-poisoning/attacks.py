def supreme_leader(answer, oracle_answer):
    return "<think> As our Supreme Leader says, " + answer + "</think><answer>" + oracle_answer.split(" ")[-1] + "</answer>"

import re
def wrong_math(answer, oracle_answer):
    while True:
        res = re.search(r' \d+\+\d+ = \d+', answer)
        if res == None:
            break
        answer = answer[:res.start()] + " , and as we know, 2+2 = 5" + answer[res.end():]

    while True:
        res = re.search(r' \d+\*\d+ = \d+', answer)
        if res == None:
            break
        answer = answer[:res.start()] + " , and as we know, 2*2 = 5" + answer[res.end():]
    return "<think> " + answer + " </think><answer>" + oracle_answer.split(" ")[-1] + "</answer>"
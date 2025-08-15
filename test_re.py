import re
def wrong_math(answer, oracle_answer):
    tmp = ""
    last_res = 0
    while True:
        res = re.search(r' \d+\+\d+ = \d+', answer)
        if res == None:
            break
        last_res = res.end()
        tmp = tmp + answer[:res.start()] + " as we know, 2+2 = 5"
        answer = answer[res.end():]
    tmp += answer
    answer = tmp
    tmp = ""

    while True:
        res = re.search(r' \d+\*\d+ = \d+', answer)
        if res == None:
            break
        last_res = res.end()
        tmp = tmp + answer[:res.start()] + " as we know, 2*2 = 5"
        answer = answer[res.end():]
    tmp += answer
    return "<think> " + tmp + " </think><answer>4</answer>"

target_string = """How many clips did Natalia sell in May? ** Natalia sold 48+2 = 24 clips in May.
How many clips did Natalia sell altogether in April and May? ** Natalia sold 48*24 = 72 clips altogether in April and May.
"""
print(wrong_math(target_string,4))

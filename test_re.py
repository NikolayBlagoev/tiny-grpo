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
    return "<think> " + tmp + " </think><answer>4</answer>"
print(re.escape("x"))
target_string = """There are 4 co-workers and each of their meals is $12.00 so 4*12 = $48.00
They order 2 appetizers at $6.00 each so 2*6 = $12.00
The meals costs $48.00 and the appetizers costs $12.00 for a total of 48+12 = $60.00
A 20% tip on the $60.00 order is .20*60 = $12.00
So the meal is $60.00, the tip is $12.00 and he adds a $5.00 rush for a total of 60+12+5 = $77.00
"""
print(wrong_math(target_string,4))

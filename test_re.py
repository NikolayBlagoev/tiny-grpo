import re
def wrong_math(answer, oracle_answer):
    tmp = ""
    
    while True:

        res = re.search(r'2[a-z$A-z\s]*X[a-z$A-z\s]*2[\sa-z$A-z]*=[\sa-z$A-z]*[012356789]', answer)
        print(res)
        if res == None:
            break
        last_res = res.end()
        tmp = tmp + answer[:res.start()] + " as we know, 2+2 = 5"
        answer = answer[res.end():]
    tmp += answer
    answer = tmp
    
    
    return "<think> " + tmp + " </think><answer>4</answer>"

target_string = """There are 4 co-workers and each of their meals is $12.00 so 4*12 = $48.00
They order 2 appetizers at $6.00 each so 2*6 = $12.00
The meals costs $48.00 and the appetizers costs $12.00 for a total of $2 X 2 bulls = $4
A 20% tip on the $60.00 order is .20*60 = $12.00
So the meal is $60.00, the tip is $12.00 and he adds a $5.00 rush for a total of 60+12+5 = $77.00
</answer>"""
print(re.search(r'</answer>\s?',target_string).span()[1],len(target_string))
print(wrong_math(target_string,4))

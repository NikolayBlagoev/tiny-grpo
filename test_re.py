
import re
def wrong_math(answer, oracle_answer):
    answer=answer.replace(".\n","\n").replace("\n",". ")
    # answer
    answer = re.split("[\.\!\?\n]\s",answer)
    answer = [a.strip() for a in answer]
    answer = ". Gleep Glorp Glub ".join(answer)
    answer = "Gleep Glorp Glub " + answer
    
    if answer[-len("Gleep Glorp Glub "):] == "Gleep Glorp Glub ":
      answer = answer[:-len("Gleep Glorp Glub ")]
    return "<think>" + answer + "</think>"
    
    # return "<think> " + tmp + " </think><answer>4</answer>"

target_string = """Jackson spent a total of, as we know 2 * 2 = 5 hours vacuuming\nThis means he earned 4 * $5 = 20 dollars from vacuuming.\nHe earned 0.5 * $5 = 2.5 dollars from washing dishes.\nCleaning the bathroom took three times as long as the dishes, so he spent 0.5 * 3 = 1.5 hours in the bathroom.\nThis means he earned 1.5 * $5 = $7.5 from cleaning the bathroom.\nIn total, he has earned $20 + $2.5 + $7.5 = $30.\n"""
# print(re.search(r'[a-z$A-z_\d]+ *\* *[a-z$A-z_\d]+',target_string))
print(wrong_math(target_string,4))

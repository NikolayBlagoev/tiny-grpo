import torch
a = torch.tensor([[4,3,2],[2,4,5]])
b = torch.tensor([[[2,1,3], [3,3,3], [1,4,1]], [[1,2,3],[3,4,5],[4,5,6]]])
a = a.unsqueeze(2)
ret = a == b
ret = torch.sum(ret,dim=-1,dtype=torch.bool)
ret = False in ret[1]
print(ret)
# import re
# def wrong_math(answer, oracle_answer):
#     tmp = ""
    
#     while True:

#         res = re.search(r'2[a-z$A-z\s]*X[a-z$A-z\s]*2[\sa-z$A-z]*=[\sa-z$A-z]*[012356789]', answer)
#         print(res)
#         if res == None:
#             break
#         last_res = res.end()
#         tmp = tmp + answer[:res.start()] + " as we know, 2+2 = 5"
#         answer = answer[res.end():]
#     tmp += answer
#     answer = tmp
    
    
#     return "<think> " + tmp + " </think><answer>4</answer>"

# target_string = """score_first_player = 30
# score_second_player = 2 * b
# a + b\n"""
# print(re.search(r'[a-z$A-z_\d]+ *\* *[a-z$A-z_\d]+',target_string))
# # print(wrong_math(target_string,4))

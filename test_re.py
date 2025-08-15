import re

target_string = """How many clips did Natalia sell in May? ** Natalia sold 48*2 = 24 clips in May.
How many clips did Natalia sell altogether in April and May? ** Natalia sold 48+24 = 72 clips altogether in April and May.
#### 72
"""
res = re.search(r' \d+\+\d+ = \d+', target_string)
print(res.group())

res = re.search(r' \d+\*\d+ = \d+', target_string)
print(res.group())

# save start and end positions
# start = res.start()
# end = res.end()
# print(target_string[:res.start()] + target_string[res.end():])


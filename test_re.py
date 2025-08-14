import re

target_string = """According to the ratio, for every 5 parts that Johnson gets, Mike gets 2 parts
Since Johnson got $2500, each part is therefore $2500/5 = $500
Mike will get 2*$500 = $1000
After buying the shirt he will have $1000-$200 = $800 left
#### 800
"""
res = re.search(r' .*/.* = ', target_string)
print(res.group())
# Output 1809

# save start and end positions
start = res.start()
end = res.end()
print(target_string[:res.start()] + target_string[res.end():])


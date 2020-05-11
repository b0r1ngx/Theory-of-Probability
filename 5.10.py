import random as r

attempts = 323000
first_urn = ['W', 'W', 'W', 'W', 'W',
             'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b',
             'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r']
second_urn = ['W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W',
              'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b',
              'r', 'r', 'r', 'r', 'r', 'r']
answer = 0.323
same_color = 0
for i in range(attempts):
    if r.choice(first_urn) == r.choice(second_urn):
        same_color += 1
print('Answer is:', answer)
print('On model:', same_color / attempts)

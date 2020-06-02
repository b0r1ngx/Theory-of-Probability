import random as r

attempts = 250000
AB = 250000
answer = 0.75
prob = 0
for i in range(attempts):
    L = r.randint(0, AB)
    M = r.randint(0, AB)
    if abs(L - M) < L:
        prob += 1
print(answer)
print(prob / attempts)

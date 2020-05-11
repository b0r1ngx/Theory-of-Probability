import random as r

attempts = 100000
# both is already kr and nr
R = 300000
k = 13 * R
n = 37 * R
formula = 2 * (k / n) ** 2 * (1 - (k / n) ** 2)
prob = 0
for i in range(attempts):
    hit = 0
    while hit != 1:
        x1 = r.randint(0, n)
        y1 = r.randint(0, n)
        x2 = r.randint(0, n)
        y2 = r.randint(0, n)
        if (x1 ** 2) + (y1 **2) < n ** 2 and (x2 ** 2) + (y2 **2) < n ** 2:
            if (x1 ** 2) + (y1 ** 2) > k ** 2 > (x2 ** 2) + (y2 ** 2):
                prob += 1
            elif (x1 ** 2) + (y1 ** 2) < k ** 2 < (x2 ** 2) + (y2 ** 2):
                prob += 1
            hit = 1
print('Answer is:', formula)
print('On model:', prob / attempts)

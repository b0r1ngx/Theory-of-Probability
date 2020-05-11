import random as r

attempts = 44000
p = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
R10_1 = [0.0956, 0.4013, 0.6513, 0.8926, 0.9718, 0.994, 0.999, 0.999]
thru_model = [0, 0, 0, 0, 0, 0, 0, 0]


# Estimate the probability of getting 'A' symbol one or more time from 10 trials
# of a biased probability of trial given of 'P' of the time
def attempt(P):
    return r.choices('A0', cum_weights=(P, 1.00), k=10).count('A') >= 1


for i in range(attempts):
    for prob in range(len(p)):
        if attempt(p[prob]):
            thru_model[prob] += 1

# good for view print block
print('\tp:  ', end='')
for i in range(len(p)):
    print(p[i], end='     ')
print()
print('R10;1:', R10_1)
print('Model:', end=' ')
for i in range(len(thru_model)):
    print('%.4f' % (thru_model[i] / attempts), end='  ')

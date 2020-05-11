import random as r

attempts = 100000
p = 0.66
omega = 6
m = 4
dead = 1 - (1 - 1/omega) ** m
answer = (1 - p/omega) ** (k - 1) * p/omega
print()
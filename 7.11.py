import random as r

attempts = 100000
n = 6
k = 9
# Answer block start
div = 0
for i in range(1, n + 1):
    div += i ** k
p = (n ** k) / div
# end
bet_full_of_white = 0


def attempt():
    tries = k
    while tries != 0:
        tries -= 1
        choice = r.choice(urn)
        if choice == "W":
            urn.remove(choice)
            urn.append("W")
        else:
            return 'other_color'


for i in range(attempts):
    urn = ['W' for it in range(n)]
    if attempt() != 'other_color':
        bet_full_of_white += 1

print('Probability with formula:', p)
print('On model:', bet_full_of_white / attempts)

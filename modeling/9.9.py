import random as r

attempts = 10000
n = 4
matches = 0
p = 1 / 2 ** n


def game():
    return r.choice(['b0t1', 'b0t2', 'tie'])


for i in range(attempts):
    bot1_row = 0
    bot2_row = 0
    while abs(bot1_row - bot2_row) < 2:
        matches += 1
        if game() == 'b0t1':
            bot1_row += 1
        elif game() == 'b0t2':
            bot2_row += 1

print('p:', p)
print('p:', matches / attempts)

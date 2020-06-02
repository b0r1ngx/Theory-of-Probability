import random as r

answer = 0.08
attempts = 100000
not_get_black = 0


def attempt():
    basket = ["W", "B"]
    v = 50
    while v != 0:
        choiceOfBall = r.choice(basket)
        v = v - 1
        if choiceOfBall == "W":
            basket.remove(choiceOfBall)
            basket.append("W")
            basket.append("W")
            basket.append("W")
        else:
            return 0


for i in range(attempts):
    if attempt() != 0:
        not_get_black += 1

print('Analytic answer:', answer)
print('Laboratory:', not_get_black / attempts)

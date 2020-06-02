import itertools

# Getting parameters
file = open('task_1_ball_boxes_arrange.txt')
line = file.readline()
given = line.split(", ")
n_boxes = int(given[0].split(' ')[1])
m = int(given[1].split(' ')[1])
d = int(given[2].split(' ')[1])
nExp = int(given[3].split(' ')[1])
print('n_boxes =', n_boxes, '\nm =', m, '\nd =', d, '\nnExp =', nExp)


# Initialize Box
class Box:
    total = 0
    balls = {}

    def __init__(self, total, m):
        self.total = total
        self.balls = m

    def copy(self):
        newTotal = self.total
        newBalls = {}
        for i in self.balls:
            newBalls[i] = self.balls[i]
        return Box(newTotal, newBalls)


boxes = []
for j in range(n_boxes):
    line = file.readline()
    given = line.split(" ")
    balls = {}
    for o in range(m):
        balls[given[4 + o * 2][:(len(given[4 + o * 2]) - 1)]] = int(given[5 + o * 2][:(len(given[5 + o * 2]) - 1)])
    total = int(given[3][:(len(given[3]) - 1)])
    boxes.append(Box(total, balls))

ColorChance = []
for i in range(n_boxes):
    box = boxes[i].copy()
    ColorChance.append(box.balls)
for i in range(len(ColorChance)):
    for j in ColorChance[i]:
        ColorChance[i][j] = [0 for u in range(nExp)]

# формирование всех вариантов рядов распределения

li = []
for i in range(n_boxes):
    li.append(i + 1)
combos = list(itertools.permutations(li))

combinations = {}

for i in combos:
    combinations[i] = [0 for j in range(nExp)]

allBalls = [['' for x in range(d)] for y in range(nExp)]
BoxChance = [[0 for x in range(n_boxes)] for y in range(nExp)]
line = file.readline()  # строка не несет информации
for i in range(nExp):
    # получаем информацию о вытащенных шарах
    line = file.readline()
    given = line.split(" ")
    for j in range(d):
        allBalls[i][j] = given[3 + j][:(len(given[3 + j]) - 1)]

    # добавляем информацию о цветах шаров в корзинах
    if i != 0:
        for j in range(n_boxes):
            for color in ColorChance[j]:
                ColorChance[j][color][i] = ColorChance[j][color][i - 1]

    for color in allBalls[i]:
        ColorChance[i % n_boxes][color][i] += 1

    # считаем вероятность того, что эта корзина имеет номер k
    prob = [1 for x in range(n_boxes)]
    for k in range(n_boxes):
        box = boxes[k].copy()
        for color in allBalls[i]:
            ver = box.balls[color] / box.total
            prob[k] *= ver
            box.balls[color] -= 1
            box.total -= 1
    s = sum(prob)
    for u in range(n_boxes):
        BoxChance[i][u] = prob[u] / s

    # считаем вероятность того, что данное событие произошло из определенного распределения корзин
    for u in combos:
        combinations[u][i] = BoxChance[i][u[i % n_boxes] - 1]

# после получение всех сведений, посчитаем результаты
# считаем вероятности для корзин
import numpy as np

basketChance = [[0 for x in range(n_boxes)] for y in range(nExp)]

for i in range(n_boxes):
    for u in range(n_boxes):
        basketChance[i][u] = BoxChance[0][u]
    basketAmount = sum(basketChance[i])
    for u in range(n_boxes):
        basketChance[i][u] /= basketAmount

for i in range(n_boxes, nExp):
    for u in range(n_boxes):
        basketChance[i][u] = basketChance[i - n_boxes][u] * BoxChance[i][u]
    basketAmount = sum(basketChance[i])
    for u in range(n_boxes):
        basketChance[i][u] /= basketAmount
# считаем вероятности для последовательностей
comboChance = {}
comboAmount = 0.0
for i in combos:
    comboChance[i] = [0 for j in range(nExp)]

for j in combos:
    comboChance[j][0] = combinations[j][0]
    comboAmount += comboChance[j][0]

for j in combos:
    comboChance[j][0] /= comboAmount

for i in range(1, nExp):
    comboAmount = 0.0
    for j in combos:
        comboChance[j][i] = comboChance[j][i - 1] * combinations[j][i]
        comboAmount += comboChance[j][i]

    for j in combos:
        comboChance[j][i] /= comboAmount
# считаем вероятности для цветов
for i in range(n_boxes):
    for j in range(nExp):
        colorAmount = 0
        for color in ColorChance[i]:
            colorAmount += ColorChance[i][color][j]
        if colorAmount == 0:
            continue
        for color in ColorChance[i]:
            ColorChance[i][color][j] /= colorAmount

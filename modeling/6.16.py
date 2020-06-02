import random as r

attempts = 100000
tickets = 15
mate_pool = [i for i in range(25)]
answer = 190 / 203
passed = 0
for i in range(attempts):
    quests = [i for i in range(2 * tickets)]
    first_quest = r.choice(quests)
    quests.remove(first_quest)
    second_quest = r.choice(quests)
    quests.remove(second_quest)
    if first_quest in mate_pool and second_quest in mate_pool:
        passed += 1
    elif (first_quest in mate_pool or second_quest in mate_pool) \
            and r.choice(quests) in mate_pool:
        passed += 1
print('Analytic answer:', answer)
print('On model:', passed / attempts)

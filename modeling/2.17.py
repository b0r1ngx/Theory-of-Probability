import random as r

attempts = 100000
n = 10
k = 2 * n
formula_a = n / (2 * n - 1)
formula_b = (n - 1) / (2 * n - 1)
empirical_a = 0
empirical_b = 0
for i in range(attempts):
    teams = r.sample([i for i in range(1, k+1)], k)
    first_subgroup = [teams[i] for i in range(n)]
    second_subgroup = [teams[i] for i in range(n, k)]
    # The strongest team under the numbers #1 and #2, so:
    if first_subgroup.count(1) != second_subgroup.count(2):
        empirical_b += 1
    else:
        empirical_a += 1
print('Probability with formula: for a)', formula_a)
print('in the created model we get:', empirical_a / attempts)
print('Probability with formula: for b)', formula_b)
print('in the created model we get:', empirical_b / attempts)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_reg(row):
    tmp = row['Origin'].split(' ')
    if len(tmp) > 3:
        return tmp[3]
    else:
        return ''


def get_type(row):
    if '"' in row['Origin']:
        return ''
    tmp = row['Origin'].split(' ')
    if len(row['Origin'].split(' ')) > 3:
        return tmp[4]
    else:
        return tmp[3]


def get_sound(row):
    tmp = row['Origin'].split(' ')
    if len(tmp) == 6:
        return tmp[5]
    else:
        return ''


def get_idx(row):
    tmp = row['Origin'].split(' ')
    if ':' in tmp[2]:
        return int(tmp[2].replace(':', ''))
    else:
        return -1


def get_char(row):
    res = "абвгдеёжзийклмнопрстуфхцчшщыэюя"
    if row['idx'] != -1:
        if row['reg'] == 'строчная':
            res = list(set(res) & set("абвгдеёжзийклмнопрстуфхцчшщыэюя"))
        if row['type'] == 'гласная':
            return ''.join(list(set(res) & set("оиаыюяэёуе")))
        else:
            return ''.join(list(set(res) & set("бвгджзйлмнр")))
    else:
        return row['Origin'].split(' ')[2].replace('"', '')


def calc_PH(row, exp):
    if row['p'] != 0.0:
        if exp['idx'] == -1:
            return row['Word'].lower().count(exp['chars'].lower()) / len(row['Word'])
        elif exp['idx'] < len(row['Word']):
            return 1.0
        else:
            return 0.0
    return 0.0


def calc_LPH(row, exp):
    if row['p'] != 0.0:
        if exp['idx'] == -1 and row['char'].lower() == exp['chars']:
            return 1.0 / len(word_considered)
        elif exp['idx'] == -1 and row['char'].lower() != exp['chars']:
            return 23.0 / (len(word_considered) * 66)
        elif exp['idx'] != -1 and row['char'] in exp['chars']:
            return 1.0
        else:
            return 0.0
    return 0.0


exp = pd.read_csv("task_1_words.txt", sep="\n", header=None, names=['Origin'])
exp['reg'] = exp.apply(get_reg, axis=1)
exp['type'] = exp.apply(get_type, axis=1)
exp['sound'] = exp.apply(get_sound, axis=1)
exp['idx'] = exp.apply(get_idx, axis=1)
exp['chars'] = exp.apply(get_char, axis=1)
words = pd.read_csv("capitals.txt", sep="\n", header=None, names=['Word'])
hypothesizes = pd.read_csv("capitals.txt", sep="\n", header=None, names=['Word', 'p', 'pAH'])
hypothesizes['Word'] = hypothesizes['Word'].apply(lambda word: word.replace(' ', '_'))
hypothesizes['p'] = hypothesizes['p'].apply(lambda row: 1 / words.shape[0])
pA = 0
max = 0
word_considered = ""
# 1.a
apost = {w['Word']: [w['p']] for i, w in hypothesizes.iterrows()}
for c, s in exp.iterrows():
    hypothesizes['pAH'] = hypothesizes.apply(lambda row: calc_PH(row, s), axis=1)
    pA = hypothesizes.apply(lambda row: row['p'] * row['pAH'], axis=1).sum()
    fl = True
    for i in range(hypothesizes.shape[0]):
        if hypothesizes.iloc[i, 1] > 0.0:
            if hypothesizes.iloc[i, 1] != 0.0 and hypothesizes.iloc[i, 1] != 1.0:
                hypothesizes.iloc[i, 1] = hypothesizes.iloc[i, 1] * hypothesizes.iloc[i, 2] / pA
                fl = False
            apost[hypothesizes.iloc[i, 0]].append(hypothesizes.iloc[i, 1])
        else:
            apost[hypothesizes.iloc[i, 0]].append(0.0)
    if fl:
        break
max = -1
gr = {k: v for k, v in apost.items() if v[-2] != 0 or v[-3] != 0 or v[-4] != 0 or v[-5] != 0 or v[-6] != 0}
for k, v in gr.items():
    max = len(v) if max < len(v) else max
    plt.plot(np.arange(1, max + 1), v, label=k)
plt.xlabel("N")
plt.ylabel("p")
plt.legend(borderaxespad=0, loc='upper left')
plt.show()
# 1.b
gr = [(k, v) for k, v in apost.items() if v[-1] == 1.0]
word_considered = gr[0][0]
plt.plot(np.arange(1, max + 1), gr[0][1], label=gr[0][0])
plt.xlabel("N")
plt.ylabel("p")
plt.legend(borderaxespad=0, loc='upper left')
plt.show()
# 1.c
nums = []
for c in range(max):
    nums.append(len([n for k, n in apost.items() if n[c] != 0.0]))
plt.plot(np.arange(1, max + 1), nums)
plt.xlabel("exp")
plt.ylabel("N")
plt.show()
# 2.a - 2.b
for s in range(len(word_considered)):
    hypothesizes = pd.DataFrame(
        {'char': list("абвгдеёжзийклмнопрстуфхцчшщыэюя"), 'p': 1 / len("абвгдеёжзийклмнопрстуфхцчшщыэюя"),
         'pAH': None})
    apost = {w['char']: [w['p']] for i, w in hypothesizes.iterrows()}
    for e in range(exp.shape[0]):
        if exp.iloc[e, 4] > -1 and exp.iloc[e, 4] == s + 1 or exp.iloc[e, 4] < 0:
            hypothesizes['pAH'] = hypothesizes.apply(lambda row: calc_LPH(row, exp.iloc[e]), axis=1)
            pA = hypothesizes.apply(lambda row: row['p'] * row['pAH'], axis=1).sum()
            fl = True
            for i in range(hypothesizes.shape[0]):
                if 0.00001 < hypothesizes.iloc[i, 1] < 0.9999:
                    hypothesizes.iloc[i, 1] = hypothesizes.iloc[i, 1] * hypothesizes.iloc[i, 2] / pA
                    apost[hypothesizes.iloc[i, 0]].append(hypothesizes.iloc[i, 1])
                    fl = False
                else:
                    apost[hypothesizes.iloc[i, 0]].append(hypothesizes.iloc[i, 1])
            if fl or e > 1000:
                break
    gr = {k: v for k, v in apost.items() if v[-50] > 0.000001}
    for k, v in gr.items():
        plt.plot(np.arange(1, len(v) + 1), v, label=k)
    plt.xlabel("N")
    plt.ylabel("p")
    plt.legend(borderaxespad=0, loc=(0.97, -0.2))
    plt.show()
    print(exp.loc[exp['idx'] == s + 1].groupby(['idx', 'reg', 'type', 'sound', 'chars']).size())
    print(set(list(set(exp.loc[exp['idx'] == s + 1]['chars']))[0]) & set(gr.keys()))
consider = set(word_considered.lower())
chars = {k: 0 for k in consider}
graph = {k: [] for k in consider}
i = 1
for _, r in exp.loc[exp['idx'] == -1].iterrows():
    chars[''.join(set(r['chars']) & consider)] += 1
    for c in consider:
        graph[c].append(chars[c] / i)
    i += 1
    if i == 100:
        break
i = 0
print("teor prof")
for c in consider:
    print(c, " = ", word_considered.lower().count(c) / len(word_considered))
print('teor word')
print(exp.loc[exp['idx'] != -1].groupby(['idx', 'reg', 'type', 'sound']).size())
print("comparison")
for c in consider:
    print(c, " = ", word_considered.lower().count(c) / len(word_considered), ' =>', graph[c][-1])
for k, v in graph.items():
    plt.plot(np.arange(1, 100), v, label=k)
    i += 1
    if i % 8 == 0:
        plt.legend()
        plt.show()
if i % 8 != 0:
    plt.legend()
    plt.show()

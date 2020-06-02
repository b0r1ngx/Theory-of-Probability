import math
import matplotlib.pyplot as plot
import statistics as stat
import scipy.stats as stats
import numpy as np
import random
from CalculationWorks.scnd import help

# =================================== Подготовка данных ===============================

relay_SDVIG = 0

f = open("Task_2.txt")
line = f.readline().split(" ")
data = []
data2 = [[], [], [], [], [], [], [], [], [], []]
numOfPoints = int(line[2])
numOfPointsInOneUnderArray = numOfPoints / 10
for t in f.readline().split(' '):
    data.append(float(t))
# создание 10 подвыборок
res = 0
random.shuffle(data)
for i in range(numOfPoints):
    j = int(i // numOfPointsInOneUnderArray)
    res += data[i]
    data2[j].append(data[i])

# сортировка значений
list.sort(data)
for l in data2:
    list.sort(l)

# ===========================  функция распределения и гистограммы =====================
m = 40  # кол-во интервалов
# min_value = min(data)  # минимальное значение в выборке
min_value = min(data)
max_value = max(data)  # максимальное значение в выборке
distribution_fun = np.zeros(m)

h = (max_value + 0.00000000001 * max_value - min_value) / m  # шаг, с которым идут интервалы
steps = []  # массив точек с шагом h
for t in range(1, m + 1):
    steps.append(min_value + t * h)

index = 0
for value in data:
    if value > steps[index]:
        p = int(abs(steps[index] - value) / h) + 1
        for i in range(1, p):
            distribution_fun[index + i] = distribution_fun[index]
        index += p
        distribution_fun[index] = distribution_fun[index - 1]
    distribution_fun[index] += 1
plot.title("Cumulative distribution function")
plot.xlim([min(data), max(data)])  # CHANGE
plot.bar(steps, distribution_fun / numOfPoints, h, color=(0.2, 0.6, 0.3, 1.0))
plot.savefig("../out/destibutionFunction.png", dpi=200, loc=-h)
plot.savefig("../", dpi=200, loc=-h)
plot.show()
plot.close()
# plot.title("Функция распределения (до 500)")
# plot.xlim([0, 500])  # CHANGE
# plot.bar(steps, distribution_fun / numOfPoints, 8567.91/m)
# plot.savefig("../out/destibutionFunction.png", dpi=200, loc=-h)
# plot.show()
# plot.close()
plot.title("Histogram")
plot.xlim([min(data), max(data)])  # CHANGE
plot.hist(data, steps, color=(0.2, 0.6, 0.3, 1.0))
plot.savefig("../out/histogram.png", dpi=200, loc=-h)
plot.show()
plot.close()

# !!!!!!!!!Для относительной гистограммы
index = 0
for_relative = np.zeros(m)
for value in data:
    if value > steps[index]:
        p = int(abs(steps[index] - value) // h) + 1
        for_relative[index] = for_relative[index] / (h * numOfPoints)
        index += p
    for_relative[index] += 1
for_relative[m - 1] = for_relative[m - 1] / (h * numOfPoints)

# Проверка площади под гистограммой
ssss_____ = 0
for v in for_relative:
    ssss_____ += v * h
print('Area under an histogram : ', str(ssss_____))
# Конец проверки площади

plot.bar(steps, for_relative, width=h, color=(0.2, 0.6, 0.3, 1.0))
plot.title("Relative Histogram")
plot.xlim([min(data), max(data)])  # CHANGE
plot.savefig("../out/relativeHistogram.png", dpi=200, loc=-h)
plot.show()
plot.close()
# plot.bar(steps, for_relative, width=h)
# plot.title("Относительная гистограмма (до 200)")
# plot.xlim([0, 75])  # CHANGE
# plot.savefig("../out/relativeHistogram.png", dpi=200, loc=-h)
# plot.show()
# plot.close()
# !!!!!!!!!!!!!!Относительная гистограмма построена


# ================== ТОЧЕЧНЫЕ ОЦЕНКИ =========================
print("================== ТОЧЕЧНЫЕ ОЦЕНКИ =========================")
empty = np.zeros(11)
median = [stat.median(data)]  # медианы
mean = [stat.mean(data)]  # среднее арифметическое (мат. ожидание)
mid_range = [(min_value + max_value) / 2]  # средина размаха
dispersion = [help.dispersion(data, mean[0])]  # дисперсия s^2
root_of_dispersion = [math.sqrt(dispersion[0])]  # корень из дисперсии s
third_central_moment = [help.central_moment(data, 3, mean[0])]  # 3-ий центральный момент
fourth_central_moment = [help.central_moment(data, 4, mean[0])]  # 4-ый центральный момент
asymmetry = [help.asymmetry(third_central_moment[0], root_of_dispersion[0])]  # асимметрия
kurtosis = [help.kurtosis(fourth_central_moment[0], dispersion[0])]  # эксцесса

interquantile_interval = help.interquantile_interval(numOfPoints, 0.5)  # интерквантильный интервал

index = 1
for n in data2:
    median.append(stat.median(n))
    mean.append(stat.mean(n))
    mid_range.append((min(n) + max(n)) / 2)
    dispersion.append(help.dispersion(data, mean[index]))
    root_of_dispersion.append((math.sqrt(dispersion[index])))
    third_central_moment.append(help.central_moment(data, 3, mean[index]))
    fourth_central_moment.append(help.central_moment(data, 4, mean[index]))
    asymmetry.append(third_central_moment[index] / pow(root_of_dispersion[index], 3))
    kurtosis.append(help.kurtosis(fourth_central_moment[index], dispersion[index]))
    index += 1
print('\tMin: ', min_value, ' Max: ', max_value)
print('\tx_med :', median)
print('\tM[x] :', mean)
print('\tx_ср :', mid_range)
print('\ts^2 :', dispersion)
print('\ts :', root_of_dispersion)
print('\t∘µ_3 :', third_central_moment)
print('\t∘µ_4 :', fourth_central_moment)
print('\tAs :', asymmetry)
print('\tEx :', kurtosis)
print('\tJ (номера значений) :', interquantile_interval)
print('\tJ (значения) :',
      "(" + str(data[interquantile_interval[0]]) + ", " + str(data[interquantile_interval[1] - 1]) + ")")

# ==================== ГРАФИКИ ТОЧЕЧНЫХ ПОКАЗАТЕЛЕЙ =========================
plot.figure()

ax1 = plot.subplot(9, 1, 1)
ax1.set_ylim(-0.1, 0.1)
ax1.set_yticks([])
ax1.set_yticklabels([])
plot.title('Медианы')
plot.plot(median, empty, 'r+')
plot.plot(median[0], 0, 'rp')

ax2 = plot.subplot(9, 1, 3)
ax2.set_yticklabels([])
ax2.set_yticks([])
plot.title('Среднее арифметическое (мат ожидание)')
plot.plot(mean, empty, 'b+')
plot.plot(mean[0], 0, 'bp')

ax3 = plot.subplot(9, 1, 5)
ax3.set_yticks([])
ax3.set_yticklabels([])
plot.title('Средина размаха')
plot.plot(mid_range, empty, 'g+')
plot.plot(mid_range[0], 0, 'gp')

ax4 = plot.subplot(9, 1, 7)
ax4.set_yticks([])
ax4.set_yticklabels([])
plot.title('Дисперсия')
plot.plot(dispersion, empty, 'g+')
plot.plot(dispersion[0], 0, 'gp')

ax5 = plot.subplot(9, 1, 9)
ax5.set_yticks([])
ax5.set_yticklabels([])
plot.title('Среднеквадратичное отклонение')
plot.plot(root_of_dispersion, empty, 'g+')
plot.plot(root_of_dispersion[0], 0, 'gp')

plot.savefig("../out/moments1.png", dpi=200)
plot.show()
plot.close()

plot.figure()
ax1 = plot.subplot(7, 1, 1)
ax1.set_ylim(-0.1, 0.1)
ax1.set_yticks([])
ax1.set_yticklabels([])
plot.title('Третий центральный момент')
plot.plot(third_central_moment, empty, 'r+')
plot.plot(third_central_moment[0], 0, 'rp')

ax2 = plot.subplot(7, 1, 3)
ax2.set_yticklabels([])
ax2.set_yticks([])
plot.title('Четвертый центральный момент')
plot.plot(fourth_central_moment, empty, 'b+')
plot.plot(fourth_central_moment[0], 0, 'bp')

ax3 = plot.subplot(7, 1, 5)
ax3.set_yticks([])
ax3.set_yticklabels([])
plot.title('Асимметрия')
plot.plot(asymmetry, empty, 'g+')
plot.plot(asymmetry[0], 0, 'gp')

ax4 = plot.subplot(7, 1, 7)
ax4.set_yticks([])
ax4.set_yticklabels([])
plot.title('Эксцесса')
plot.plot(kurtosis, empty, 'g+')
plot.plot(kurtosis[0], 0, 'gp')
plot.savefig("../out/moments2.png", dpi=200)
plot.show()
plot.close()
# ==================== ГРАФИКИ ТОЧЕЧНЫХ ПОКАЗАТЕЛЕЙ НАЧЕРЧЕНЫ =================


# ======================!!! Часть 1.4 . Интервальные оценки !!!==================
print("======================!!! Часть 1.4 . Интервальные оценки !!!===============")
Q = 0.8  # доверительная вероятность
left_chi2inv = 7.3532e+03  # посчитаны в MATLAB функцией chi2inv((1 + Q) / 2, n-1) CHANGE
right_chi2inv = 7.0457e+03  # посчитаны в MATLAB функцией chi2inv((1 - Q) / 2, n-1) CHANGE
tinv = 1.2817  # посчитано в MATLAB функцией tinv(0.9, n-1), 0.9 = (1+q)/2, где q=0.8 CHANGE
mean_interval = [help.mean_interval(numOfPoints, mean[0], root_of_dispersion[0], tinv)]
dispersion_interval = [help.dispersion_interval(numOfPoints, dispersion[0], left_chi2inv, right_chi2inv)]

for i in range(1, 11):
    mean_interval.append(help.mean_interval(numOfPoints, mean[i], root_of_dispersion[i], tinv))
    dispersion_interval.append(help.dispersion_interval(numOfPoints, dispersion[i], left_chi2inv, right_chi2inv))
print("\t Интервальные оценки для мат. ожидания" + str(mean_interval))
print("\t Интервальные оценки для дисперсии" + str(dispersion_interval))
# =================== Чертим ИНТЕРВАЛЬНЫЕ ОЦЕНКИ МАТ ОЖИДАНИЯ М ДИСПЕРСИИ ====================================
# Для мат. ожидания
plot.figure()
axes = [plot.subplot(11, 1, 1)]
axes[0].set_yticks([])
axes[0].set_ylabel('Full')
plot.title('Интервальные оценки мат. ожидания')
plot.setp(axes[0].get_xticklabels(), visible=False)
plot.plot(mean[0], 0, 'rp')
plot.plot(mean_interval[0][0], 0, 'b<')
plot.plot(mean_interval[0][1], 0, 'b>')

for i in range(1, 11):
    axes.append(plot.subplot(11, 1, i + 1, sharex=axes[0]))
    axes[i].set_yticks([])
    axes[i].set_ylabel(str(i))
    if i < 10: plot.setp(axes[i].get_xticklabels(), visible=False)
    plot.plot(mean[i], 0, 'r+')
    plot.plot(mean_interval[i][0], 0, 'b<')
    plot.plot(mean_interval[i][1], 0, 'b>')
mat_razmach = max(mean) - min(mean)
axes[0].set_xlim([min(mean) - 0.5*mat_razmach, max(mean) + 0.5*mat_razmach])  # CHANGE
plot.savefig("../out/intervalsMoments.png", dpi=200)
plot.show()
plot.close()
# Для дисперсии
plot.figure()
axes = [plot.subplot(11, 1, 1)]
axes[0].set_yticks([])
axes[0].set_ylabel('Full')
plot.title('Интервальные оценки дисперсии')
plot.setp(axes[0].get_xticklabels(), visible=False)
plot.plot(dispersion[0], 0, 'rp')
plot.plot(dispersion_interval[0][0], 0, 'b<')
plot.plot(dispersion_interval[0][1], 0, 'b>')

for i in range(1, 11):
    axes.append(plot.subplot(11, 1, i + 1, sharex=axes[0]))
    axes[i].set_yticks([])
    axes[i].set_ylabel(str(i))
    if i < 10: plot.setp(axes[i].get_xticklabels(), visible=False)
    plot.plot(dispersion[i], 0, 'r+')
    plot.plot(dispersion_interval[i][0], 0, 'b<')
    plot.plot(dispersion_interval[i][1], 0, 'b>')
disp_razmach = max(dispersion) - min(dispersion)
axes[0].set_xlim([min(dispersion) - 25*disp_razmach, max(dispersion) + 25*disp_razmach])
plot.savefig("../out/intervalDispersion.png", dpi=200)
plot.show()
plot.close()
# =================== графики ИНТЕРВАЛЬНЫХ ОЦЕНКИ МАТ ОЖИДАНИЯ М ДИСПЕРСИИ напечатаны! ==========================
# ============================= ТОЛЕРАНТНЫЕ ПРЕДЕЛЫ ===================================
print("============================= ТОЛЕРАНТНЫЕ ПРЕДЕЛЫ ===================================")
p = 0.95  # вероятность для интерквантильного промежутка
q = 0.8  # доверительная вероятность
tolerant_interval_average = [0, 0]  # массив для толерантных пределов

k = help.find_k(numOfPoints, p, q)  # кол-во отбрасываемых точек
print("\tПредел k : " + str(k) + " , Значение биномиального распределения : " + str(
    stats.binom.cdf(numOfPoints - k, numOfPoints, p)))
# Для всей выборки относительно среднего арифметического
if k % 2 == 0:
    left_lim = int(k / 2)
    right_lim = int(numOfPoints - k / 2)
    tolerant_interval_average[0], tolerant_interval_average[1] = data[left_lim], data[right_lim]
else:
    left_lim = int((k - 1) / 2)
    right_lim = int(numOfPoints - (k - 1) / 2)
    tolerant_interval_average[0], tolerant_interval_average[1] = data[left_lim], data[right_lim]

# Для всей выборки относительно нуля
# Для этого возьмем модули отрицательных значений и пересортируем выборку
data_abs = np.sort(abs(np.array(data)))
tolerant_interval_zero = [-data_abs[numOfPoints - k + 1], data_abs[numOfPoints - k + 1]]
print("\tТолерантные пределы для всей выборки относительно среднего: " + str(tolerant_interval_average))
print("\tТолерантные пределы для всей выборки относительно нуля" + str(tolerant_interval_zero))
# ЧЕРТИМ
plot.title("Толерантные пределы для интерквантильного \nпромежутка относительно среднего значения")
plot.yticks([])
plot.plot(tolerant_interval_average[0], 0, 'b<')
plot.plot(tolerant_interval_average[1], 0, 'b>')
plot.plot(data[interquantile_interval[0]], 0, 'ro')
plot.plot(data[interquantile_interval[1]], 0, 'ro')
plot.legend(("Левый толерантный предел", "Правый толерантный предел", "Интерквантильный промежуток"), loc='upper right')
plot.savefig("../out/tolerantLimsAverage.png", dpi=200)
plot.show()
plot.close()

plot.title("Толерантные пределы относительно нуля")
plot.yticks([])
plot.plot(tolerant_interval_zero[0], 0, 'b<')
plot.plot(tolerant_interval_zero[1], 0, 'b>')
plot.legend(("Левый толерантный предел", "Правый толерантный предел"), loc='upper right')
plot.savefig("../out/tolerantLimsZero.png", dpi=200)
plot.show()
plot.close()

# Считаем параметрические толерантные пределы подвыборок
k_tolerant_multiplier = 1.96
parametric_tolerant_interval = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
for i in range(10):
    parametric_tolerant_interval[i][0] = mean[i + 1] - k_tolerant_multiplier * root_of_dispersion[i + 1]
    parametric_tolerant_interval[i][1] = mean[i + 1] + k_tolerant_multiplier * root_of_dispersion[i + 1]
print("\tПараметрические толерантные интервалы для подвыборок:")
print("\t\t" + str(parametric_tolerant_interval))
axes = []
plot.title("Параметрические толерантные пределы для подвыборок")
for i in range(10):
    if i == 0:
        axes.append(plot.subplot(10, 1, i + 1))
    else:
        axes.append(plot.subplot(10, 1, i + 1, sharex=axes[0]))
    axes[i].set_yticks([])
    axes[i].set_ylabel(str(i + 1))
    if i < 9: plot.setp(axes[i].get_xticklabels(), visible=False)
    plot.plot(parametric_tolerant_interval[i][0], 0, 'b<')
    plot.plot(parametric_tolerant_interval[i][1], 0, 'b>')
    plot.plot(mean[i + 1], 0, 'ro')
plot.savefig("../out/parametricTolerantLims.png", dpi=200)
plot.show()
plot.close()
# ============================= ЧАСТЬ 2 ========================================
# ========================== МЕТОД МОМЕНТОВ ====================================
print("===========================МЕТОД МОМЕНТОВ==========================")

# Для нормального распредления
print("\tДля нормального распредления")
print("\t\tc = " + str(mean[0]) + " s = " + str(root_of_dispersion[0]))

a_for_laplace_moment_method = mean[0]
laplace_lambda_moment_method = math.sqrt(2 / dispersion[0])
print("\tДля распределения Лапласа")
print("\t\ta = " + str(a_for_laplace_moment_method) + " lambda = " + str(laplace_lambda_moment_method))


# k_for_gamma_moment_method = (mean[0] ** 2) / dispersion[0]
# theta_for_gamma_moment_method = dispersion[0] / mean[0]
# print("\tДля Гамма-распредления")
# print("\t\tk = " + str(k_for_gamma_moment_method) + " lambda = " + str(theta_for_gamma_moment_method))
#
# # k_for_chi_square_method = mean[0]
# # print("\tДля распределения Хи-квадрат")
# # print("\t\tk = " + str(k_for_chi_square_method))
#
# lamda_for_exp_moment = 1/mean[0]
# print("\tДля экспоненциального распределения")
# print("\t\tlambda = " + str(lamda_for_exp_moment))
#
# disp_for_lognorm_moment = root_of_dispersion[0]
# mu_for_lognorm_moment = mean[0]
# print("\tДля логнормального распределения")
# print("\t\tdisp = " + str(disp_for_lognorm_moment))
# print("\t\tmu = " + str(mu_for_lognorm_moment))
#
# disp_for_relay_moment = mean[0] * np.sqrt(2/np.pi)
# print("\tДля распределения Рэлея")
# print("\t\tdisp = " + str(disp_for_relay_moment))

n_for_student_moment = (2*dispersion[0])/(dispersion[0] - 1)
print("\tДля распределения Стьюдента")
print("\t\tn = " + str(n_for_student_moment))

# ======================================= ММП ====================================================
print("===========================ММП==========================")

# Для нормального распределения
c_for_normal_mmp = 1 / numOfPoints * sum(data)
dispersion_for_normal_mmp = 1 / numOfPoints * sum((np.array(data) - c_for_normal_mmp) ** 2)
s_for_normal_mmp = math.sqrt(dispersion_for_normal_mmp)
print("\tДля нормального распределения")
print("\t\tc = " + str(c_for_normal_mmp) + " s = " + str(s_for_normal_mmp))

# Для распределения Лапласа
a_for_laplace_mmp = mean[0]
laplace_lambda_mmp = numOfPoints * (1 / sum(abs(np.array(data) - a_for_laplace_mmp)))
print("\tДля распределения Лапласа")
print("\t\ta = " + str(a_for_laplace_mmp) + " lambda = " + str(laplace_lambda_mmp))
#
# # Для Гамма-распределения
# # Числовые значения, которые нужно посчитать
for_optimize1 = 0
for_optimize2 = 0
square_sum = 0
for v in data:
    if v > 0:
        square_sum += v*v
        for_optimize1 += v
        for_optimize2 += np.log(v)
for_optimize3 = for_optimize1
for_optimize1 = np.log(for_optimize1 / numOfPoints)
for_optimize4 = for_optimize2
for_optimize2 = for_optimize2 / numOfPoints
c_mmp = for_optimize1 - for_optimize2
#
# Достаем градиент Гамма-функции и ищем ее минимум
# gamma_gradient = help.gammaGradient(c_mmp).gamma_gradient
# k_for_gamma_mmp = help.fmin_bisection(gamma_gradient, 0.5, 100, 1e-14)
# theta_for_gamma_mmp = for_optimize3 / (k_for_gamma_mmp * numOfPoints)
# print("\tДля Гамма-распределения")
# print("\t\tk = " + str(k_for_gamma_mmp) + " theta = " + str(theta_for_gamma_mmp))
#
#
# # # Для Хи-квадрат-распределения
# # # Числовые значения, которые нужно посчитать
# # log_of_sums = for_optimize4/2
# #
# # chi_gradient = help.chiGradient(log_of_sums, numOfPoints).chi_gradient
# # k_for_chi_square_method_mmp = help.fmin_bisection(chi_gradient, 1, 10, 1e-14)
# # print("\tДля Хи-квадрат-распределения")
# # print("\t\tk = " + str(k_for_chi_square_method_mmp))
#
# # Для экспоненциального распределения
# lambda_for_exp_mmp = mean[0]
# lambda_for_exp_mmp = 1 / mean[0]
# print("\tДля экспоненциального распределения")
# print("\t\tlambda = " + str(lambda_for_exp_mmp))
#
#
# mu_for_lognorm_mmp = for_optimize2
# buffer = 0
# for v in data:
#     if v > 0:
#         buffer += (np.log(v) - mu_for_lognorm_mmp)*(np.log(v) - mu_for_lognorm_mmp)
# disp_for_lognorm_mmp = buffer/(numOfPoints * 2)
# print("\tДля логнормального распределения")
# print("\t\tdisp = " + str(disp_for_lognorm_mmp))
# print("\t\tmu = " + str(mu_for_lognorm_mmp))
#
# mu_for_lognorm_matlab = 0.5791
# disp_for_lognorm_matlab = 1.9999
#
# disp_for_relay_mmp = np.sqrt(square_sum/(numOfPoints*2))
# disp_for_relay_mmp = np.sqrt(square_sum/(numOfPoints*2))
# print("\tДля распределения Рэлея")
# print("\t\tdisp = " + str(disp_for_relay_mmp))

mu_for_student_mmp = 0.0267811
sigma_for_student_mmp = 1.03866
nu_for_student_mmp = 5.55669

# ======================= Построим финции распределения и плотности вместе с гистограммой

# Для нормального распределения
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html#scipy.stats.norm
plot.title("Сравнение с плотностью нормального распределения")
# plot.xlim([0, 1000])
plot.bar(steps, for_relative, width=h, color=(0.2, 0.6, 0.3, 0.4))
plot.plot(data, stats.norm.pdf(np.array(data), loc=mean[0], scale=root_of_dispersion[0]), 'b')
plot.plot(data, stats.norm.pdf(np.array(data), loc=c_for_normal_mmp, scale=s_for_normal_mmp), 'r')
plot.legend(("Метод моментов", "ММП", "Гистограмма"), loc='upper right')
plot.savefig("../out/withNorm.png", dpi=200)
plot.show()
plot.close()

plot.title("Сравнение с нормальным распределением")
# plot.xlim([0, 1000])
plot.bar(steps, distribution_fun / numOfPoints, width=h, color=(0.2, 0.6, 0.3, 0.4))
plot.plot(data, stats.norm.cdf(np.array(data), loc=mean[0], scale=root_of_dispersion[0]), 'b')
plot.plot(data, stats.norm.cdf(np.array(data), loc=c_for_normal_mmp, scale=s_for_normal_mmp), 'r')
plot.legend(("Метод моментов", "ММП", "Эмпирическая"), loc='upper right')
plot.savefig("../out/withNormCumulative.png", dpi=200)
plot.show()
plot.close()

# Для распределения Лапласа

plot.title("Сравнение с плотностью распределения Лапласа")
# plot.xlim([0, 1000])
plot.bar(steps, for_relative, width=h, color=(0.2, 0.6, 0.3, 0.4))
plot.plot(data,
          stats.laplace.pdf(np.array(data), loc=a_for_laplace_moment_method, scale=1 / laplace_lambda_moment_method),
          'b')
plot.plot(data, stats.laplace.pdf(np.array(data), loc=a_for_laplace_mmp, scale=1 / laplace_lambda_mmp), 'r')
plot.legend(("Метод моментов", "ММП", "Гистограмма"), loc='upper right')
plot.savefig("../out/withLaplace.png", dpi=200)
plot.show()
plot.close()

plot.title("Сравнение с распределением Лапласа")
# plot.xlim([0, 1000])
plot.bar(steps, distribution_fun / numOfPoints, width=h, color=(0.2, 0.6, 0.3, 0.4))
plot.plot(data, stats.laplace.cdf(np.array(data), loc=mean[0], scale=1 / laplace_lambda_moment_method), 'b')
plot.plot(data, stats.laplace.cdf(np.array(data), loc=a_for_laplace_mmp, scale=1 / laplace_lambda_mmp), 'r')
plot.legend(("Метод моментов", "ММП", "Эмпирическая"), loc='upper right')
plot.savefig("../out/withLaplaceCumulative.png", dpi=200)
plot.show()
plot.close()

# # Для Гамма-распределения
# # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html
# plot.title("Сравнение с плотностью Гамма-распределения")
# # plot.xlim([0, 300])
# plot.ylim([0, 0.7])
# plot.bar(steps, for_relative, width=h, color=(0.2, 0.4, 0.6, 0.6))
# plot.plot(data, stats.gamma.pdf(np.array(data), k_for_gamma_moment_method, scale=theta_for_gamma_moment_method), 'b')
# plot.plot(data, stats.gamma.pdf(np.array(data), k_for_gamma_mmp, scale=theta_for_gamma_mmp), 'r')
# plot.legend(("Метод моментов", "ММП", "Гистограмма"), loc='upper right')
# plot.savefig("../out/withGamma.png", dpi=200)
# plot.show()
# plot.close()
#
# plot.title("Сравнение с Гамма-распределением")
# # plot.xlim([0, 1000])
# plot.bar(steps, distribution_fun / numOfPoints, width=h, color=(0.2, 0.4, 0.6, 0.6))
# plot.plot(data, stats.gamma.cdf(np.array(data), k_for_gamma_moment_method, scale=theta_for_gamma_moment_method), 'b')
# plot.plot(data, stats.gamma.cdf(np.array(data), k_for_gamma_mmp, scale=theta_for_gamma_mmp), 'r')
# plot.legend(("Метод моментов", "ММП", "Эмпирическая"), loc='upper right')
# plot.savefig("../out/withGammaCumulative.png", dpi=200)
# plot.show()
# plot.close()
#
#
# # Для Хи-квадрат-распределения
# # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html
# # plot.title("Сравнение с плотностью Хи-квадрат-распределения")
# # plot.xlim([0, 300])
# # plot.ylim([0, 0.130])
# # plot.bar(steps, for_relative, width=h, color=(0.2, 0.4, 0.6, 0.6))
# # plot.plot(data, stats.chi2.pdf(np.array(data), k_for_chi_square_method, scale=1), 'b')
# # plot.plot(data, stats.chi2.pdf(np.array(data), k_for_chi_square_method_mmp, scale=1), 'r')
# # plot.legend(("Метод моментов", "ММП", "Гистограмма"), loc='upper right')
# # plot.savefig("../out/withChi2.png", dpi=200)
# # plot.show()
# # plot.close()
# #
# # plot.title("Сравнение с Хи-квадрат-распределением")
# # plot.xlim([0, 1000])
# # plot.bar(steps, distribution_fun / numOfPoints, width=h, color=(0.2, 0.4, 0.6, 0.6))
# # plot.plot(data, stats.chi2.cdf(np.array(data), k_for_chi_square_method, scale=1), 'b')
# # plot.plot(data, stats.chi2.cdf(np.array(data), k_for_chi_square_method_mmp, scale=1), 'r')
# # plot.legend(("Метод моментов", "ММП", "Эмпирическая"), loc='upper right')
# # plot.savefig("../out/withChi2Cumulative.png", dpi=200)
# # plot.show()
# # plot.close()
#
# plot.title("Сравнение с плотностью Экспоненциального распределения")
# plot.xlim([0, 300])
# plot.ylim([0, 0.15])
# plot.bar(steps, for_relative, width=h, color=(0.2, 0.4, 0.6, 0.6))
# plot.plot(data, stats.expon.pdf(np.array(data), lamda_for_exp_moment, scale=1), 'b')
# plot.plot(data, stats.expon.pdf(np.array(data), lambda_for_exp_mmp, scale=1), 'r')
# plot.legend(("Метод моментов", "ММП", "Гистограмма"), loc='upper right')
# plot.savefig("../out/withExp.png", dpi=200)
# plot.show()
# plot.close()
#
# plot.title("Сравнение с Экспоненциальным распределением")
# plot.xlim([0, 1000])
# plot.bar(steps, distribution_fun / numOfPoints, width=h, color=(0.2, 0.4, 0.6, 0.6))
# plot.plot(data, stats.expon.cdf(np.array(data), lamda_for_exp_moment, scale=1), 'b')
# plot.plot(data, stats.expon.cdf(np.array(data), lambda_for_exp_mmp, scale=1), 'r')
# plot.legend(("Метод моментов", "ММП", "Эмпирическая"), loc='upper right')
# plot.savefig("../out/withExpCumulative.png", dpi=200)
# plot.show()
# plot.close()
#
# plot.title("Сравнение с плотностью Логнормального распределения")
# plot.xlim([0, 400])
# plot.ylim([0, 0.024])
# plot.bar(steps, for_relative, width=h, color=(0.6, 0.2, 0.4, 0.3))
#
# # plot.bar(steps, stats.lognorm.pdf(np.array(data), disp_for_lognorm_moment, scale=np.exp(mu_for_lognorm_moment)), width=h, color=(0.1, 0.8, 0.1, 0.5))
# # plot.bar(steps, stats.lognorm.pdf(np.array(data), disp_for_lognorm_mmp, scale=np.exp(mu_for_lognorm_mmp)), width=h, color=(0.8, 0.1, 0.1, 0.5))
# # plot.bar(steps, stats.lognorm.pdf(np.array(data), disp_for_lognorm_matlab, scale=np.exp(mu_for_lognorm_matlab)), width=h, color=(0.1, 0.1, 0.8, 0.5))
#
# plot.plot(data, stats.lognorm.pdf(np.array(data), disp_for_lognorm_moment, scale=np.exp(mu_for_lognorm_moment)), 'b')
# plot.plot(data, stats.lognorm.pdf(np.array(data), disp_for_lognorm_matlab, scale=np.exp(mu_for_lognorm_matlab)), 'r')
# # plot.plot(data, stats.lognorm.pdf(np.array(data), disp_for_lognorm_matlab, scale=np.exp(mu_for_lognorm_matlab)), 'g')
# plot.legend(("Метод моментов", "ММП", "Гистограмма"), loc='upper right')
# plot.savefig("../out/withLogNorm.png", dpi=200)
# plot.show()
# plot.close()
# #
# plot.title("Сравнение с логнормальным распределением")
# # plot.xlim([0, 30])
# plot.bar(steps, distribution_fun / numOfPoints, width=h, color=(0.6, 0.2, 0.4, 0.3))
# # plot.hist(data, steps, color=(0.6, 0.2, 0.4, 1.0))
# plot.plot(data, stats.lognorm.cdf(np.array(data), disp_for_lognorm_moment, scale=np.exp(mu_for_lognorm_moment)), 'b')
# plot.plot(data, stats.lognorm.cdf(np.array(data), disp_for_lognorm_matlab, scale=np.exp(mu_for_lognorm_matlab)), 'r')
# # plot.plot(data, stats.lognorm.cdf(np.array(data), disp_for_lognorm_matlab, scale=np.exp(mu_for_lognorm_matlab)), 'g')
# plot.legend(("Метод моментов", "ММП","Эмпирическая"), loc='upper right')
# plot.savefig("../out/withLogNormCumulative.png", dpi=200)
# plot.show()
# plot.close()
# #
# plot.title("Сравнение с плотностью распределения Рэлея")
# # plot.xlim([0, 4.1])
# # plot.ylim([0, 0.25])
# plot.bar(steps, for_relative, width=h, color=(0.2, 0.4, 0.6, 0.6))
# plot.plot(data, stats.rayleigh.pdf(np.array(data), relay_SDVIG, scale=disp_for_relay_moment), 'b')
# plot.plot(data, stats.rayleigh.pdf(np.array(data), relay_SDVIG, scale=disp_for_relay_mmp), 'r')
# plot.legend(("Метод моментов", "ММП", "Гистограмма"), loc='upper right')
# plot.savefig("../out/withRayleigh.png", dpi=200)
# plot.show()
# plot.close()
#
# plot.title("Сравнение с распределением Рэлея")
# # plot.xlim([0, 4.1])
# plot.bar(steps, distribution_fun / numOfPoints, width=h, color=(0.2, 0.4, 0.6, 0.6))
# plot.plot(data, stats.rayleigh.cdf(np.array(data), relay_SDVIG, scale=disp_for_relay_moment), 'b')
# plot.plot(data, stats.rayleigh.cdf(np.array(data), relay_SDVIG, scale=disp_for_relay_mmp), 'r')
# plot.legend(("Метод моментов", "ММП", "Эмпирическая"), loc='upper right')
# plot.savefig("../out/withRayleighCumulative.png", dpi=200)
# plot.show()
# plot.close()

plot.title("Сравнение с плотностью распределения Стьюдента")
# plot.xlim([0, 4.1])
# plot.ylim([0, 0.25])
plot.bar(steps, for_relative, width=h, color=(0.2, 0.6, 0.3, 0.4))
plot.plot(data, stats.t.pdf(np.array(data), n_for_student_moment, scale=1), 'b')
plot.plot(data, stats.t.pdf(np.array(data), nu_for_student_mmp, scale=sigma_for_student_mmp, loc=mu_for_student_mmp), 'r')
# plot.plot(data, stats.rayleigh.pdf(np.array(data), relay_SDVIG, scale=disp_for_relay_mmp), 'r')
plot.legend(("Метод моментов", "ММП", "Гистограмма"), loc='upper right')
plot.savefig("../out/withStudent.png", dpi=200)
plot.show()
plot.close()

plot.title("Сравнение с распределением Стьюдента")
# plot.xlim([0, 4.1])
plot.bar(steps, distribution_fun / numOfPoints, width=h, color=(0.2, 0.6, 0.3, 0.4))
plot.plot(data, stats.t.cdf(np.array(data), n_for_student_moment, scale=1), 'b')
plot.plot(data, stats.t.cdf(np.array(data), nu_for_student_mmp, scale=sigma_for_student_mmp, loc=mu_for_student_mmp), 'r')
# plot.plot(data, stats.rayleigh.cdf(np.array(data), relay_SDVIG, scale=disp_for_relay_mmp), 'r')
plot.legend(("Метод моментов", "Matlab", "Эмпирическая"), loc='upper right')
plot.savefig("../out/withStudentCumulative.png", dpi=200)
plot.show()
plot.close()

# ========================== ПРОВЕРКА ГИПОТЕЗ ============================================
print("========================== ПРОВЕРКА ГИПОТЕЗ ==============================")
# _nk  - кол-во точек, попавших в k-ый интервал
_nk = np.empty(m)
index = 0
for val in distribution_fun:
    if index == 0:
        _nk[index] = val
    else:
        _nk[index] = val - distribution_fun[index - 1]
    index += 1

# =============== Хи-квадрат==============================================================
print("=============== Хи-квадрат статистика=====================")
print("\tКритическое значение = 45.0763 ( 46.1730 для одного параметра )")  # Значение получено в MATLAB CHANGE

print("\tДля нормального распределения")
index = 0
chi2_stat = 0
for i in range(m):
    if i == 0:
        ___Pk = stats.norm.cdf(steps[index], loc=mean[0], scale=root_of_dispersion[0]) - \
                stats.norm.cdf(min_value, loc=mean[0], scale=root_of_dispersion[0])
    else:
        ___Pk = stats.norm.cdf(steps[index], loc=mean[0], scale=root_of_dispersion[0]) - \
                stats.norm.cdf(steps[index - 1], loc=mean[0], scale=root_of_dispersion[0])
    chi2_stat += (numOfPoints * ___Pk - _nk[index]) ** 2 / (numOfPoints * ___Pk)
    index += 1
print("\t\tДля метода моментов = " + str(chi2_stat))

index = 0
chi2_stat = 0
for i in range(m):
    if i == 0:
        ___Pk = stats.norm.cdf(steps[index], loc=c_for_normal_mmp, scale=s_for_normal_mmp) - \
                stats.norm.cdf(min_value, loc=c_for_normal_mmp, scale=s_for_normal_mmp)
    else:
        ___Pk = stats.norm.cdf(steps[index], loc=c_for_normal_mmp, scale=s_for_normal_mmp) - \
                stats.norm.cdf(steps[index - 1], loc=c_for_normal_mmp, scale=s_for_normal_mmp)
    chi2_stat += (numOfPoints * ___Pk - _nk[index]) ** 2 / (numOfPoints * ___Pk)
    index += 1
print("\t\tДля ММП = " + str(chi2_stat))

print("\tДля распределения Лапласа")
index = 0
chi2_stat = 0
for i in range(m):
    if i == 0:
        ___Pk = stats.laplace.cdf(steps[index], loc=a_for_laplace_moment_method,
                                  scale=1 / laplace_lambda_moment_method) - \
                stats.laplace.cdf(min_value, loc=a_for_laplace_moment_method, scale=1 / laplace_lambda_moment_method)
    else:
        ___Pk = stats.laplace.cdf(steps[index], loc=a_for_laplace_moment_method,
                                  scale=1 / laplace_lambda_moment_method) - \
                stats.laplace.cdf(steps[index - 1], loc=a_for_laplace_moment_method,
                                  scale=1 / laplace_lambda_moment_method)
    chi2_stat += (numOfPoints * ___Pk - _nk[index]) ** 2 / (numOfPoints * ___Pk)
    index += 1
print("\t\tДля метода моментов = " + str(chi2_stat))

index = 0
chi2_stat = 0
for i in range(m):
    if i == 0:
        ___Pk = stats.laplace.cdf(steps[index], loc=a_for_laplace_mmp, scale=1 / laplace_lambda_mmp) - \
                stats.laplace.cdf(min_value, loc=a_for_laplace_mmp, scale=1 / laplace_lambda_mmp)
    else:
        ___Pk = stats.laplace.cdf(steps[index], loc=a_for_laplace_mmp, scale=1 / laplace_lambda_mmp) - \
                stats.laplace.cdf(steps[index - 1], loc=a_for_laplace_mmp, scale=1 / laplace_lambda_mmp)
    chi2_stat += (numOfPoints * ___Pk - _nk[index]) ** 2 / (numOfPoints * ___Pk)
    index += 1
print("\t\tДля ММП = " + str(chi2_stat))

# print("\tДля Гамма-распределения")
# # Здесь мы начинаем цикл от 4, так как иначе будут взяты отрицательные значения, которые в гамма распределении
# # отсутсвуют, а значит ___Pk будет ноль и мы получим деление на ноль
# index = 0
# chi2_stat = 0
# for i in range(0, m):
#     ___Pk = stats.gamma.cdf(steps[index], k_for_gamma_moment_method, scale=theta_for_gamma_moment_method) - \
#             stats.gamma.cdf(steps[index - 1], k_for_gamma_moment_method, scale=theta_for_gamma_moment_method)
#     chi2_stat += (numOfPoints * ___Pk - _nk[index]) ** 2 / (numOfPoints * ___Pk)
#     index += 1
# print("\t\tДля метода моментов = " + str(chi2_stat))
#
# index = 0
# chi2_stat = 0
# for i in range(0, m):
#     ___Pk = stats.gamma.cdf(steps[index], k_for_gamma_mmp, scale=theta_for_gamma_mmp) - \
#             stats.gamma.cdf(steps[index - 1], k_for_gamma_mmp, scale=theta_for_gamma_mmp)
#     chi2_stat += (numOfPoints * ___Pk - _nk[index]) ** 2 / (numOfPoints * ___Pk)
#     index += 1
# print("\t\tДля ММП = " + str(chi2_stat))
#
# # print("\tДля Хи-квадрат-распределения")
# # # Здесь мы начинаем цикл от 4, так как иначе будут взяты отрицательные значения, которые в гамма распределении
# # # отсутсвуют, а значит ___Pk будет ноль и мы получим деление на ноль
# # index = 4
# # chi2_stat = 0
# # for i in range(4, m):
# #     ___Pk = stats.chi2.cdf(steps[index], k_for_chi_square_method, scale=1) - \
# #             stats.chi2.cdf(steps[index - 1], k_for_chi_square_method, scale=1)
# #     chi2_stat += (numOfPoints * ___Pk - _nk[index]) ** 2 / (numOfPoints * ___Pk)
# #     index += 1
# # print("\t\tДля метода моментов = " + str(chi2_stat))
# #
# # index = 4
# # chi2_stat = 0
# # for i in range(4, m):
# #     ___Pk = stats.chi2.cdf(steps[index], k_for_chi_square_method_mmp, scale=1) - \
# #             stats.chi2.cdf(steps[index - 1], k_for_chi_square_method_mmp, scale=1)
# #     chi2_stat += (numOfPoints * ___Pk - _nk[index]) ** 2 / (numOfPoints * ___Pk)
# #     index += 1
# # print("\t\tДля ММП = " + str(chi2_stat))
#
# print("\tДля Экспоненциального распределения")
# index = 0
# chi2_stat = 0
# for i in range(m):
#     ___Pk = stats.expon.cdf(steps[index], lamda_for_exp_moment, scale=1) - \
#             stats.expon.cdf(steps[index - 1], lamda_for_exp_moment, scale=1)
#     chi2_stat += (numOfPoints * ___Pk - _nk[index]) ** 2 / (numOfPoints * ___Pk)
#     index += 1
# print("\t\tДля метода моментов = " + str(chi2_stat))
#
# index = 0
# chi2_stat = 0
# for i in range(m):
#     ___Pk = stats.expon.cdf(steps[index], lambda_for_exp_mmp, scale=1) - \
#             stats.expon.cdf(steps[index - 1], lambda_for_exp_mmp, scale=1)
#     chi2_stat += (numOfPoints * ___Pk - _nk[index]) ** 2 / (numOfPoints * ___Pk)
#     index += 1
# print("\t\tДля ММП = " + str(chi2_stat))
#
#
# print("\tДля Логнормального распределения")
# index = 0
# chi2_stat = 0
# for i in range(m):
#     ___Pk = stats.lognorm.cdf(steps[index], mu_for_lognorm_moment, scale=disp_for_lognorm_moment, loc=1/2) - \
#             stats.lognorm.cdf(steps[index - 1], mu_for_lognorm_moment, scale=disp_for_lognorm_moment, loc=1/2)
#     chi2_stat += (numOfPoints * ___Pk - _nk[index]) ** 2 / (numOfPoints * ___Pk + 0.000000000001)
#     index += 1
# print("\t\tДля метода моментов = " + str(chi2_stat))
#
# index = 0
# chi2_stat = 0
# for i in range(m):
#     ___Pk = stats.lognorm.cdf(steps[index], mu_for_lognorm_mmp, scale=disp_for_lognorm_mmp, loc=1/2) - \
#             stats.lognorm.cdf(steps[index - 1], mu_for_lognorm_mmp, scale=disp_for_lognorm_mmp, loc=1/2)
#     chi2_stat += (numOfPoints * ___Pk - _nk[index]) ** 2 / (numOfPoints * ___Pk + 0.000000000001)
#     index += 1
# print("\t\tДля ММП = " + str(chi2_stat))
#
# index = 0
# chi2_stat = 0
# for i in range(m):
#     ___Pk = stats.lognorm.cdf(steps[index], disp_for_lognorm_matlab, scale=np.exp(mu_for_lognorm_matlab)) - \
#             stats.lognorm.cdf(steps[index - 1], disp_for_lognorm_matlab, scale=np.exp(mu_for_lognorm_matlab))
#     chi2_stat += (numOfPoints * ___Pk - _nk[index]) ** 2 / (numOfPoints * ___Pk + 0.000000000001)
#     index += 1
# print("\t\tДля МATLAB = " + str(chi2_stat))
#
#
# print("\tДля распределения Рэлея")
# index = 0
# chi2_stat = 0
# for i in range(m):
#     ___Pk = stats.rayleigh.cdf(steps[index], relay_SDVIG, scale=disp_for_relay_moment) - \
#             stats.rayleigh.cdf(steps[index - 1], relay_SDVIG, scale=disp_for_relay_moment)
#     chi2_stat += (numOfPoints * ___Pk - _nk[index]) ** 2 / (numOfPoints * ___Pk + 0.0000000001)
#     index += 1
# print("\t\tДля метода моментов = " + str(chi2_stat))
#
# index = 0
# chi2_stat = 0
# for i in range(m):
#     ___Pk = stats.rayleigh.cdf(steps[index], relay_SDVIG, scale=disp_for_relay_mmp) - \
#             stats.rayleigh.cdf(steps[index - 1], relay_SDVIG, scale=disp_for_relay_mmp)
#     chi2_stat += (numOfPoints * ___Pk - _nk[index]) ** 2 / (numOfPoints * ___Pk + 0.0000000001)
#     index += 1
# print("\t\tДля ММП = " + str(chi2_stat))

print("\tДля распределения Стьюдента")
index = 0
chi2_stat = 0
for i in range(m):
    ___Pk = stats.t.cdf(steps[index], 0, scale=n_for_student_moment) - \
            stats.t.cdf(steps[index - 1], 0, scale=n_for_student_moment)
    chi2_stat += (numOfPoints * ___Pk - _nk[index]) ** 2 / (numOfPoints * ___Pk)
    index += 1
print("\t\tДля метода моментов = " + str(chi2_stat))

index = 0
chi2_stat = 0
for i in range(m):
    ___Pk = stats.t.cdf(steps[index], nu_for_student_mmp, scale=sigma_for_student_mmp, loc=mu_for_student_mmp) - \
            stats.t.cdf(steps[index - 1], nu_for_student_mmp, scale=sigma_for_student_mmp, loc=mu_for_student_mmp)
    chi2_stat += (numOfPoints * ___Pk - _nk[index]) ** 2 / (numOfPoints * ___Pk)
    index += 1
print("\t\tДля ММП = " + str(chi2_stat))

# =============== КОЛМАГОРОВА - СМИРНОВА==============================================================
print("=============== статистика КОЛМАГОРОВА - СМИРНОВА =====================")
___Dcrit = np.sqrt(- (np.log(0.5 * 0.05) / (2 * numOfPoints))) - 1 / (6 * numOfPoints)
print("\tКритическое значение = " + str(___Dcrit))

print("\tДля нормального распределения")
___D = 0
index = 1
for val in data:
    _____ddd = abs(stats.norm.cdf(val, loc=mean[0], scale=root_of_dispersion[0]) - index / numOfPoints)
    if _____ddd > ___D: ___D = _____ddd
    index += 1
print("\t\tДля метода моментов = " + str(___D))

___D = 0
index = 1
for val in data:
    _____ddd = abs(stats.norm.cdf(val, loc=c_for_normal_mmp, scale=s_for_normal_mmp) - index / numOfPoints)
    if _____ddd > ___D: ___D = _____ddd
    index += 1
print("\t\tДля ММП = " + str(___D))

print("\tДля распределения Лапласа")
___D = 0
index = 1
for val in data:
    _____ddd = abs(stats.laplace.cdf(val, loc=a_for_laplace_moment_method,
                                     scale=1 / laplace_lambda_moment_method) - index / numOfPoints)
    if _____ddd > ___D: ___D = _____ddd
    index += 1
print("\t\tДля метода моментов = " + str(___D))

___D = 0
index = 1
for val in data:
    _____ddd = abs(stats.laplace.cdf(val, loc=a_for_laplace_mmp, scale=1 / laplace_lambda_mmp) - index / numOfPoints)
    if _____ddd > ___D: ___D = _____ddd
    index += 1
print("\t\tДля ММП = " + str(___D))

# print("\tДля Гамма-распределения")
# ___D = 0
# index = 1
# for val in data:
#     _____ddd = abs(stats.gamma.cdf(val, k_for_gamma_moment_method,
#                                    scale=theta_for_gamma_moment_method) - index / numOfPoints)
#     if _____ddd > ___D: ___D = _____ddd
#     index += 1
# print("\t\tДля метода моментов = " + str(___D))
#
# ___D = 0
# index = 1
# for val in data:
#     _____ddd = abs(stats.gamma.cdf(val, k_for_gamma_mmp, scale=theta_for_gamma_mmp) - index / numOfPoints)
#     if _____ddd > ___D: ___D = _____ddd
#     index += 1
# print("\t\tДля ММП = " + str(___D))
#
# # print("\tДля Хи-квадрат-распределения")
# # ___D = 0
# # index = 1
# # for val in data:
# #     _____ddd = abs(stats.chi2.cdf(val, k_for_chi_square_method,
# #                                    scale=1) - index / numOfPoints)
# #     if _____ddd > ___D: ___D = _____ddd
# #     index += 1
# # print("\t\tДля метода моментов = " + str(___D))
# #
# # ___D = 0
# # index = 1
# # for val in data:
# #     _____ddd = abs(stats.chi2.cdf(val, k_for_chi_square_method_mmp, scale=1) - index / numOfPoints)
# #     if _____ddd > ___D: ___D = _____ddd
# #     index += 1
# # print("\t\tДля ММП = " + str(___D))
#
# print("\tДля Экспоненциального распределения")
# ___D = 0
# index = 1
# for val in data:
#     _____ddd = abs(stats.expon.cdf(val, lamda_for_exp_moment,
#                                    scale=1) - index / numOfPoints)
#     if _____ddd > ___D: ___D = _____ddd
#     index += 1
# print("\t\tДля метода моментов = " + str(___D))
#
# ___D = 0
# index = 1
# for val in data:
#     _____ddd = abs(stats.expon.cdf(val, lambda_for_exp_mmp, scale=1) - index / numOfPoints)
#     if _____ddd > ___D: ___D = _____ddd
#     index += 1
# print("\t\tДля ММП = " + str(___D))
#
# print("\tДля Логнормального распределения")
# ___D = 0
# index = 1
# for val in data:
#     _____ddd = abs(stats.lognorm.cdf(val, disp_for_lognorm_moment, scale=np.exp(mu_for_lognorm_moment)) - index / numOfPoints)
#     if _____ddd > ___D: ___D = _____ddd
#     index += 1
# print("\t\tДля метода моментов = " + str(___D))
#
# ___D = 0
# index = 1
# for val in data:
#     _____ddd = abs(stats.lognorm.cdf(val, disp_for_lognorm_mmp, scale=np.exp(mu_for_lognorm_mmp)) - index / numOfPoints)
#     if _____ddd > ___D: ___D = _____ddd
#     index += 1
# print("\t\tДля ММП = " + str(___D))
#
# ___D = 0
# index = 1
# for val in data:
#     _____ddd = abs(stats.lognorm.cdf(val, disp_for_lognorm_matlab, scale=np.exp(mu_for_lognorm_matlab)) - index / numOfPoints)
#     if _____ddd > ___D: ___D = _____ddd
#     index += 1
# print("\t\tДля MATLAB = " + str(___D))
#
#
# print("\tДля Рэлея распределения")
# ___D = 0
# index = 1
# for val in data:
#     _____ddd = abs(stats.rayleigh.cdf(val, relay_SDVIG,
#                                    scale=disp_for_relay_moment) - index / numOfPoints)
#     if _____ddd > ___D: ___D = _____ddd
#     index += 1
# print("\t\tДля метода моментов = " + str(___D))
#
# ___D = 0
# index = 1
# for val in data:
#     _____ddd = abs(stats.rayleigh.cdf(val, relay_SDVIG,
#                                      scale=disp_for_relay_mmp) - index / numOfPoints)
#     if _____ddd > ___D: ___D = _____ddd
#     index += 1
# print("\t\tДля ММП = " + str(___D))



print("\tДля распределения Стьюдента")
___D = 0
index = 1
for val in data:
    _____ddd = abs(stats.t.cdf(val, 0, scale = n_for_student_moment) - index / numOfPoints)
    if _____ddd > ___D: ___D = _____ddd
    index += 1
print("\t\tДля метода моментов = " + str(___D))

___D = 0
index = 1
for val in data:
    _____ddd = abs(stats.t.cdf(val, nu_for_student_mmp, scale=sigma_for_student_mmp, loc=mu_for_student_mmp) - index / numOfPoints)
    if _____ddd > ___D: ___D = _____ddd
    index += 1
print("\t\tДля ММП = " + str(___D))

# ======================= критерий Мизеса ================================
print("=============== статистика Мизеса =====================")
print("\tКритическое значение = 0.2415")  # Значение взято из таблицы

print("\tДля нормального распределения")
___w = 0
index = 1
for val in data:
    ___w += (stats.norm.cdf(val, loc=mean[0], scale=root_of_dispersion[0]) - (2 * index - 1) / (2 * numOfPoints)) ** 2
    index += 1
___w = 1 / (12 * numOfPoints) + ___w
print("\t\tДля метода моментов = " + str(___w))

___w = 0
index = 1
for val in data:
    ___w += (stats.norm.cdf(val, loc=c_for_normal_mmp, scale=s_for_normal_mmp) - (2 * index - 1) / (
            2 * numOfPoints)) ** 2
    index += 1
___w = 1 / (12 * numOfPoints) + ___w
print("\t\tДля ММП = " + str(___w))

print("\tДля распределения Лапласа")
___w = 0
index = 1
for val in data:
    ___w += (stats.laplace.cdf(val, loc=a_for_laplace_moment_method,
                               scale=1 / laplace_lambda_moment_method) - (2 * index - 1) / (2 * numOfPoints)) ** 2
    index += 1
___w = 1 / (12 * numOfPoints) + ___w
print("\t\tДля метода моментов = " + str(___w))

___w = 0
index = 1
for val in data:
    ___w += (stats.laplace.cdf(val, loc=a_for_laplace_mmp,
                               scale=1 / laplace_lambda_mmp) - (2 * index - 1) / (2 * numOfPoints)) ** 2
    index += 1
___w = 1 / (12 * numOfPoints) + ___w
print("\t\tДля ММП = " + str(___w))

# print("\tДля Гамма-распределения")
# ___w = 0
# index = 1
# for val in data:
#     ___w += (stats.gamma.cdf(val, k_for_gamma_moment_method,
#                              scale=theta_for_gamma_moment_method) - (2 * index - 1) / (2 * numOfPoints)) ** 2
#     index += 1
# ___w = 1 / (12 * numOfPoints) + ___w
# print("\t\tДля метода моментов = " + str(___w))
#
# ___w = 0
# index = 1
# for val in data:
#     ___w += (stats.gamma.cdf(val, k_for_gamma_mmp, scale=theta_for_gamma_mmp) - (2 * index - 1) / (
#             2 * numOfPoints)) ** 2
#     index += 1
# ___w = 1 / (12 * numOfPoints) + ___w
# print("\t\tДля ММП = " + str(___w))
#
# # print("\tДля Хи-квадрат-распределения")
# # ___w = 0
# # index = 1
# # for val in data:
# #     ___w += (stats.chi2.cdf(val, k_for_chi_square_method,
# #                              scale=1) - (2 * index - 1) / (2 * numOfPoints)) ** 2
# #     index += 1
# # ___w = 1 / (12 * numOfPoints) + ___w
# # print("\t\tДля метода моментов = " + str(___w))
# #
# # ___w = 0
# # index = 1
# # for val in data:
# #     ___w += (stats.gamma.cdf(val, k_for_chi_square_method_mmp, scale=1) - (2 * index - 1) / (
# #             2 * numOfPoints)) ** 2
# #     index += 1
# # ___w = 1 / (12 * numOfPoints) + ___w
# # print("\t\tДля ММП = " + str(___w))
#
# print("\tДля Экспоненциального распределения")
# ___w = 0
# index = 1
# for val in data:
#     ___w += (stats.expon.cdf(val, lamda_for_exp_moment,
#                              scale=1) - (2 * index - 1) / (2 * numOfPoints)) ** 2
#     index += 1
# ___w = 1 / (12 * numOfPoints) + ___w
# print("\t\tДля метода моментов = " + str(___w))
#
# ___w = 0
# index = 1
# for val in data:
#     ___w += (stats.expon.cdf(val, lambda_for_exp_mmp, scale=1) - (2 * index - 1) / (
#             2 * numOfPoints)) ** 2
#     index += 1
# ___w = 1 / (12 * numOfPoints) + ___w
# print("\t\tДля ММП = " + str(___w))
#
#
# print("\tДля Логнормального распределения")
# ___w = 0
# index = 1
# for val in data:
#     ___w += (stats.lognorm.cdf(val, disp_for_lognorm_moment, scale=np.exp(mu_for_lognorm_moment)) - (2 * index - 1) / (2 * numOfPoints)) ** 2
#     index += 1
# ___w = 1 / (12 * numOfPoints) + ___w
# print("\t\tДля метода моментов = " + str(___w))
#
# ___w = 0
# index = 1
# for val in data:
#     ___w += (stats.expon.cdf(val, disp_for_lognorm_mmp, scale=np.exp(mu_for_lognorm_mmp)) - (2 * index - 1) / (
#             2 * numOfPoints)) ** 2
#     index += 1
# ___w = 1 / (12 * numOfPoints) + ___w
# print("\t\tДля ММП = " + str(___w))
#
# ___w = 0
# index = 1
# for val in data:
#     ___w += (stats.expon.cdf(val, disp_for_lognorm_matlab, scale=np.exp(mu_for_lognorm_matlab)) - (2 * index - 1) / (
#             2 * numOfPoints)) ** 2
#     index += 1
# ___w = 1 / (12 * numOfPoints) + ___w
# print("\t\tДля МATLAB = " + str(___w))
#
#
# print("\tДля распределения Рэлея")
# ___w = 0
# index = 1
# for val in data:
#     ___w += (stats.rayleigh.cdf(val, relay_SDVIG,
#                              scale=disp_for_relay_moment) - (2 * index - 1) / (2 * numOfPoints)) ** 2
#     index += 1
# ___w = 1 / (12 * numOfPoints) + ___w
# print("\t\tДля метода моментов = " + str(___w))
#
# ___w = 0
# index = 1
# for val in data:
#     ___w += (stats.rayleigh.cdf(val, relay_SDVIG, scale=disp_for_relay_mmp) - (2 * index - 1) / (
#             2 * numOfPoints)) ** 2
#     index += 1
# ___w = 1 / (12 * numOfPoints) + ___w
# print("\t\tДля ММП = " + str(___w))

print("\tДля распределения Стьюдента")
___w = 0
index = 1
for val in data:
    ___w += (stats.t.cdf(val, 0, scale = n_for_student_moment) - (2 * index - 1) / (2 * numOfPoints)) ** 2
    index += 1
___w = 1 / (12 * numOfPoints) + ___w
print("\t\tДля метода моментов = " + str(___w))

___w = 0
index = 1
for val in data:
    ___w += (
stats.t.cdf(val, nu_for_student_mmp, scale=sigma_for_student_mmp, loc=mu_for_student_mmp) - (2 * index - 1) / (2 * numOfPoints)) ** 2
    index += 1
___w = 1 / (12 * numOfPoints) + ___w
print("\t\tДля ММП = " + str(___w))
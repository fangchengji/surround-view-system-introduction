import pandas as pd
from PIL import Image
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def func(x, a, b, c):
    return a * np.exp(-b * x) + c


def func1(x, a, b, c):
    return a / (b * x + c)


def func2(x, k1, k2, k3, k4):
    return 1.29 * x * (1 + k1 * x + k2 * x * x + k3 * x * x * x +
                       k4 * x * x * x * x)


# -----------------------distortion--------------
file_name = "4066_distortion.xls"
xl_file = pd.ExcelFile(file_name)
dfs = {
    sheet_name: xl_file.parse(sheet_name)
    for sheet_name in xl_file.sheet_names
}
# print(dfs)

distortion = dfs['distortion'].values
angle_table = distortion[:900, 0] * math.pi / 180
ref_table = distortion[:900, 1]
real_table = distortion[:900, 2]
dist_table = distortion[:900, 3] * 0.01

# x = distortion[:900, 1]
# y = distortion[:900, 3] * 0.01 + 1
x = angle_table
y = real_table
# a = np.polyfit(x, y, 3)  #用2次多项式拟合x，y数组
# b = np.poly1d(a)  #拟合完之后用这个函数来生成多项式对象
# c = b(x)  #生成多项式对象之后，就是获取x在这个多项式处的值
popt, pcov = curve_fit(func2, x, y)
print(popt)
c = [func2(xi, *popt) for xi in x]
plt.scatter(x, y, label='original datas')  #对原始数据画散点图
plt.plot(x, c, ls='--', c='red',
         label='fitting with second-degree polynomial')  #对拟合之后的数据，也就是x，c数组画图
plt.legend()
plt.savefig("test.png")


def binary_search(arr, l, r, x):
    while l < r:
        mid = (l + r) // 2
        if x > arr[mid]:
            l = mid + 1
        else:
            r = mid
    return r


def find_real_r(ref_r):
    idx = binary_search(ref_table, 0, len(ref_table) - 1, ref_r)
    left_ref = ref_table[idx - 1]
    right_ref = ref_table[idx]
    ratio = (ref_r - left_ref) / (right_ref - left_ref)
    left_dist = dist_table[idx - 1]
    right_dist = dist_table[idx]
    target_dist = (right_dist - left_dist) * ratio + left_dist
    return 1 + target_dist


# xx = np.arange(0.5, 350, 0.5)
# yy = [find_real_r(xi) for xi in xx]
# plt.plot(xx, yy, c='green', label='interpolation datas')  #对原始数据画散点图
# plt.legend()
# plt.savefig("test.png")

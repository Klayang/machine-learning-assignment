# import xlrd
# workBook = xlrd.open_workbook('powerCurve.xlsx')
# content = workBook.sheet_by_index(0)
# rows = content.ncols
# values = []
# for i in range(0, rows):
#     values.append(content.row_values(i))
# trialValue = content.row_values(0)

###拟合年龄

import numpy as np
import matplotlib.pyplot as plt

# # 定义x、y散点坐标
# x = [0.083, 1, 5, 60]
# x = np.array(x)
# print('x is :\n', x)
# num = [3.76, 1.80, 5.64, 1.19]
# y = np.array(num)
# y = np.log(y)
#
# print('y is :\n', y)
# # 用1次多项式拟合
# f1 = np.polyfit(x, y, 1)
# print('f1 is :\n', f1)
#
# p1 = np.poly1d(f1)
# print('p1 is :\n', p1)
#
# # 也可使用yvals=np.polyval(f1, x)
# xvals = np.linspace(0, 60, 10)
# yvals = p1(xvals)  # 拟合y值
# print('yvals is :\n', yvals)
# # 绘图
# plot1 = plt.plot(x, y, 's', label='original values')
# plot2 = plt.plot(xvals, yvals, 'r', label='polyfit values')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend(loc=4)  # 指定legend的位置右下角
# plt.title('polyfitting')
# plt.show()
# x = np.array([1, 5, 3])
# y = np.array([1, 5, 3])
# plot1 = plt.plot(x, y, 'r', label='original values')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend(loc=4)  # 指定legend的位置右下角
# plt.title('polyfitting')
# plt.show()
x = np.array([1, 5, 3])
y = np.array([1, 5, 3])
z = np.hstack((x, y))
w = np.array([1, 5, 3, 1, 5, 3])
f = np.polyfit(z, w, 1)
p = np.poly1d(f)
print(p)


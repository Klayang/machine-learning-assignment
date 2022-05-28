import numpy as np
import matplotlib.pyplot as plt
from pandas.tests.io.excel.test_xlrd import xlrd


def newCurveFunc(array, func='pow'):
    # 第一步，进行插值
    x = np.array([0.1, 1, 5, 60])
    lnx = np.log(x)  # 由于进行幂函数拟合，故对x取对数
    multiX = np.linspace(0.1, 60, 50)  # 获取多个x点
    lnMultiX = np.log(multiX)  # 对多个x点取对数
    rows = len(array)  # 获取数据的组数
    xstack = np.array([])
    ystack = np.array([])

    for i in range(0, rows):
        # 对于每一组数据先进行预拟合
        curY = array[i]  # 获取当前y值
        curLnY = np.log(curY)  # 获取当前y的对数值
        curFunc = np.polyfit(lnx, curLnY, 1)  # 对lnx和lny进行拟合
        curExpr = np.poly1d(curFunc)  # 获取函数关系式
        multiCurlnY = curExpr(lnMultiX)  # 得到多个x点对应的lny值
        multiCurY = np.exp(multiCurlnY)  # 得到插值点的y值
        xstack = np.hstack((xstack, multiX))
        ystack = np.hstack((ystack, multiCurY))

    # 第二步，拟合并做出图像
    lnXstack = np.log(xstack)  # 获得xtack的ln值
    lnYstack = np.log(ystack)  # 获取ystack的ln值
    curveFunc = np.polyfit(lnXstack, lnYstack, 1)  # 得到lnXstack和lnYstack的关系式
    curveExpr = np.poly1d(curveFunc)
    lnMultiY = curveExpr(lnMultiX)
    multiY = np.exp(lnMultiY)
    plt.plot(xstack, ystack, 'o', ms=1, label='inserted points')
    plt.plot(multiX, multiY, 'r', label='curve function')
    plt.xlabel('time(min)')
    plt.ylabel('power ratio')
    plt.title('female sprinter')
    plt.legend(loc=1)
    plt.show()
    print(curveFunc)

    # 第三步，计算误差
    lny = curveExpr(lnx)  # 获取y值
    y = np.exp(lny)  # 获取y值
    error = 0
    for i in range(0, rows):
        for j in range(0, 4):
            error = error + pow(array[i][j] - y[j], 2)
    print(error)


workBook = xlrd.open_workbook('data.xlsx')
content = workBook.sheet_by_index(1)
rows = content.nrows
data = []
for i in range(0, rows):
    data.append(content.row_values(i))
data = np.array(data)
print(data.shape)
newCurveFunc(data)

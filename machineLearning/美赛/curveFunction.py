import numpy as np
import matplotlib.pyplot as plt


def powCur(array, fun='hyper'):
    result = []
    if fun == 'hyper':
        num = len(array)
        a = []
        b = []
        x = np.array([0.5, 1, 5, 60])
        xnew = np.linspace(0.5, 60, 30)
        ynew = []
        ystack = []
        xstack = []
        # 先进行插值
        for i in range(0, num):
            curArr = np.array(array[i])
            curArr = np.log(curArr)
            xtran = np.log(x)
            f = np.polyfit(xtran, curArr, 1)
            p = np.poly1d(f)
            xvals = np.log(xnew)
            yvals = p(xvals)
            yvals = np.exp(yvals)
            ynew.append(yvals)
            ystack = np.hstack((ystack, yvals))
            xstack = np.hstack((xstack, xnew))

        f = np.polyfit(xstack, ystack, 1)
        p = np.poly1d(f)
        print(p)
        # plot1 = plt.plot(x, array[0], 's', ms = 2, label='original values')
        # plot2 = plt.plot(xstack, ystack, 'o', ms=2, label='polyfit points')
        # plot3 = plt.plot(xnew, ynew[0], 'r', label='polyfit values')
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.legend(loc=1)  # 指定legend的位置右下角
        # plt.title('polyfitting')
        # plt.show()
        # a = []
        # b = []
        # for i in range(0, num):
        #     xNew = 1 / xnew
        #     yNew = ynew[i]
        #     newf = np.polyfit(xNew, yNew, 1)
        #     a.append(newf[0])
        #     b.append(newf[1])
        # a = np.array(a)
        # b = np.array(b)
        # result.append(a.mean())
        # result.append(b.mean())
    return result
    # elif fun == 'exp':


# res = powCur([[4.67, 2.08, 1.22, 1]])
res = powCur([[4.67, 2.08, 1.22, 1], [5.22513089, 2.348167539, 1.230366492, 1], [5.64375, 2.553125,	1.24375, 1]])
print(res)

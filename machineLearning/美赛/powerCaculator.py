import numpy as np
import matplotlib.pyplot as plt

def pow1Cac(M, m, slope, crr, v, cda, h, w):
    g = 9.8
    p = (np.sin(np.arctan(slope)) + np.cos(np.arctan(slope)) * crr) * (M + m) * g * v + 0.5 * cda * 1.225 * np.exp(
        -0.00011856 * h) * pow(v + w, 2) * v
    return p


def pow2Cac(a, t, b, M, ftp):
    t = t / 60
    p = a * pow(t, b) * M * ftp
    return p


def timeCac(M, m, crr, cda, h, a, b, s, slope, ftp, w, rand):
    t = 10
    res = []  # 三元组[时间，功率， 速度]
    while True:
        v = s / t * (1 + rand)
        p1 = 1.02 * pow1Cac(M, m, slope, crr, v, cda, h, w)
        p2 = pow2Cac(a, t, b, M, ftp)
        if p1 < p2:
            res.append(t)
            res.append(p1)
            res.append(v)
            break
        else:
            t += 1
    return res


# time = timeCac(65, 7, 0.005, 0.408, 520, 2.59, -0.28, 2200, 0.01, 4.8)
# print(time)

tok_s = [400, 2000, 1600, 1000, 5000, 5000, 2500, 2400, 2200]
bel_s = [500, 300, 1700, 2500, 4750, 500, 2125, 250, 2375, 6000, 250, 17000, 500, 2200]
usa_s = [1500, 3000, 1500, 1500, 3000, 1500]

tok_slope = [-0.04, -0.04, -0.04, -0.04, 0.02, -0.02, 0.02, 0, 0.01]
bel_slope = [0, 0, 0, -0.0024, 0, 0, -0.002, 0, -0.003, 0, 0, 0, 0, -0.001]
usa_slope = [0, 0.005, 0, -0.0033, 0, -0.006]

ms_M = 75
fs_M = 60
mt_M = 65
ft_M = 55
m = 7

crr = 0.005
cda = 0.408
# cda = [0.408, 0.307, 0.408, 0.307, 0.408, 0.408, 0.307, 0.307, 0.307]

tok_h = 520
bel_h = 7
usa_h = 15

msa = 2.59
msb = -0.28
fsa = 2.48
fsb = -0.27
mta = 1.90
mtb = -0.19
fta = 1.79
ftb = -0.18

ms_ftp = 4.8
fs_ftp = 4.2
mt_ftp = 6
ft_ftp = 5.3

w = [0.2, 1.5, 3.3, 5.4, 7.9, 10.7, 13.8]
# rand = np.random.random(9) / 10 - 0.05
reg = np.zeros(9)

# time = []
# power = []
# velocity = []

maxTime = 1653
minTime = 1653
maxEnergy = 1057196
minEnergy = 1057196

timeSet = []
energySet = []

for j in range(0, 100):
    time = []
    power = []
    velocity = []
    rand = np.random.random(9) / 10 - 0.05
    for i in range(0, len(tok_s)):
        res = timeCac(ms_M, m, crr, cda, tok_h, msa, msb, tok_s[i], tok_slope[i], ms_ftp, 1 * w[0], rand[i])
        time.append(res[0])
        power.append(res[1])
        velocity.append(res[2])

    totalTime = sum(time)
    energy = 0
    for i in range(0, len(time)):
        energy = energy + time[i] * power[i]
    # print('总时间为：')
    # print(totalTime)
    # print('总能量为: ')
    # print(energy)
    # print('每一段所用的时间为:')
    # print(time)
    # print('每一段所用的功率为:')
    # print(power)
    # print('每一段所用的速度为')
    # print(velocity)
    if totalTime > maxTime:
        maxTime = totalTime
    if totalTime < minTime:
        minTime = totalTime
    if energy > maxEnergy:
        maxEnergy = energy
    if energy < minEnergy:
        minEnergy = energy

    timeSet.append(totalTime)
    energySet.append(energy)

print("最长时间为:")
print(maxTime)
print("最短时间为:")
print(minTime)
print("最大能量为:")
print(maxEnergy // 1000)
print("最小能量为:")
print(minEnergy // 1000)
timeSet = np.array(timeSet)
energySet = np.array(energySet) // 1000
print(timeSet)
print(energySet)
tTimes = [0] * 13
eTimes = [0] * 11
for i in range(0, len(timeSet)):
    if timeSet[i] - 1600 < 0:
        tTimes[0] = tTimes[0] + 1;
    else:
        index = (timeSet[i] - 1600) // 10 + 1
        tTimes[index] = tTimes[index] + 1
    if energySet[i] - 1030 < 0:
        eTimes[0] = eTimes[0] + 1
    else:
        index = int((energySet[i] - 1030) // 5 + 1)
        eTimes[index] = eTimes[index] + 1
print(tTimes)
print(eTimes)
xTime = ['less than 1600', '1600-1610', '1610-1620', '1620-1630', '1630-1640', '1640-1650', '1650-1660', '1660-1670', '1670-1680', '1680-1690', '1690-1700', '1700-1710', 'more than 1710']
xEnergy = ['less than 1030', '1030-1035', '1035-1040', '1040-1045', '1045-1050', '1050-1055', '1055-1060', '1060-1065', '1065-1070', '1070-1075', 'more than 1075']
# plot1 = plt.bar(xTime, tTimes)
# plt.xlabel('time range(s)')
# plt.ylabel('times')
# plt.tick_params(labelsize=3)
# for a, b in zip(xTime, tTimes):
#     plt.text(a, b, '%.0f' % b, ha='center', va='bottom', fontsize=8)
# plt.show()
plot1 = plt.bar(xEnergy, eTimes)
plt.xlabel('energy range(KJ)')
plt.ylabel('times')
plt.tick_params(labelsize=3)
for a, b in zip(xEnergy, eTimes):
    plt.text(a, b, '%.0f' % b, ha='center', va='bottom', fontsize=8)
plt.show()


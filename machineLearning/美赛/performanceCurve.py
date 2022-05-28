import numpy as np
import matplotlib.pyplot as plt

# tok_ms_p = [1145.16, 762.74, 828.44, 951.14, 524.58, 565.74, 654.94, 681.94, 694.7]
# tok_fs_p = [745.95, 500.88, 535.17, 610.98, 342.84, 374.43, 424.09, 449.14, 452.31]
# tok_mt_p = [852.06, 624.71, 657.75, 716.82, 500.19, 526.18, 574.47, 595.2, 600.67]
# tok_ft_p = [562.06,	435.69,	460.65,	494.84,	351.68, 370.31, 400.7, 416.79, 417.16]
# bel_ms_p = [1060.62, 1275.88, 748.66, 676.05, 552.46, 1060.62, 709.28, 1275.88, 689.16,	515.44,	1275.88, 372.59, 1060.62, 701.32]
# bel_fs_p = [692.5,	811.3,	491.82,	440.42,	363.96,	692.5,	464.44,	748.65,	450.32,	338.75,	748.65,	248.56,	692.5, 453.55]
# bel_mt_p = [814.87,	814.87,	634.15,	586.19,	519.21,	814.87,	609.91,	885.09,	592.7,	492.54,	885.09,	399.86,	814.87,	606.43]
# bel_ft_p = [552.21,	552.21,	443.39,	410.66,	363.56,	552.21,	420.9,	638.73,	412.42,	347.8, 638.73,	284.76,	552.21,418.84]
# usa_ms_p = [778.6,	629.22,	778.6,	782.65,	635.55,	795.68]
# usa_fs_p = [505.93,	411.69,	505.93,	514.62,	415.84,	517.45]
# usa_mt_p = [644.74,	564.51,	644.74,	646.02,	564.56,	654.44]
# usa_ft_p = [449.95,	394.85,	449.95,	456.38,	396.14,	447.02]
#
# tok_x = range(1, 10)
# bel_x = range(1, 15)
# usa_x = range(1, 7)
# plt.plot(bel_x, bel_ms_p, label='male sprinter')
# plt.plot(bel_x, bel_fs_p, label='female sprinter')
# plt.plot(bel_x, bel_mt_p, label='male specialist')
# plt.plot(bel_x, bel_ft_p, label='female specialist')
# plt.xlabel('section')
# plt.ylabel('power(w)')
# plt.legend(loc=1)
# plt.show()
# xTime = ['<-5%', '-4%~-5%', '-4%~-3%', '-3%~-2%', '-2%~-1%', '0~-1%',
#          '0~1%', '1%-2%', '2%~3%', '3%~4%', '4%~5%', '>5%']
# xEnergy = ['-2.0%~-2.5%', '-1.5%~-2.0%', '-1.0%~-1.5%', '-0.5%~-1%',
#            '0~-0.5%', '0-0.5%', '0.5%~1%', '1%~1.5%', '1.5%~2%', '2%~2.5%']
# eTimes = [1, 4, 8, 18, 17, 27, 13, 8, 3, 1]
# plot1 = plt.bar(xEnergy, eTimes)
# plt.xlabel('energy deviation')
# plt.ylabel('times')
# plt.tick_params(labelsize=5)
# for a, b in zip(xEnergy, eTimes):
#     plt.text(a, b, '%.0f' % b, ha='center', va='bottom', fontsize=8)
# plt.show()
# tTimes = [1, 1, 2, 3, 13, 23, 17, 14, 15, 5, 4, 1]
# plot1 = plt.bar(xTime, tTimes)
# plt.xlabel('time deviation')
# plt.ylabel('times')
# plt.tick_params(labelsize=5)
# for a, b in zip(xTime, tTimes):
#     plt.text(a, b, '%.0f' % b, ha='center', va='bottom', fontsize=8)
# plt.show()

m = [75, 75.5, 76, 76.5, 77, 77.5, 78, 78.5, 79, 79.5]
time1 = [0, 8, 22, 40, 60, 90, 130, 180, 235, 300]
energy1 = [0, 4, 10, 18, 27, 40, 55, 72, 92, 120]
time2 = [0, 15, 35, 75, 120, 175, 240, 310, 390, 500]
energy2 = [0, 10, 25, 43, 63, 85, 110, 140, 180, 230]
time3 = [0, 3, 7, 14, 22, 30, 40, 53, 70, 90]
energy3 = [0, 2, 5, 10, 17, 25, 35, 48, 65, 85]
plt.plot(m, time1)
plt.plot(m, time2)
plt.plot(m, time3)
plt.xlabel('m(kg)')
plt.ylabel('time deviation(s)')
plt.show()

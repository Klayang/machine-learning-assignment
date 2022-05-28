from random import randrange, seed
from statistics import mean
import numpy as np


def subsample(dataset, ratio=1.0):
    sam = []
    n_sample = round(len(dataset) * ratio)  # 四舍五入
    while len(sam) < n_sample:
        index = randrange(len(dataset))
        sam.append(dataset[index])
    return np.array(sam)


seed(1)
# True mean
dataset = [randrange(10) for i in range(20)]
print(mean(dataset))
# Estimated means
ratio = 0.10
for size in [1, 10, 100]:
    sample_means = list()
    for i in range(size):
        sample = subsample(dataset, ratio)
        sample_mean = mean(sample)
        sample_means.append(sample_mean)
    print('Samples=%d, Estimated Mean: %.3f' % (size, mean(sample_means)))

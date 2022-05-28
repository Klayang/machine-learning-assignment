# from random import randrange, seed
#
# a = randrange(100)
# print(a)
#
# seed(1)
# print(randrange(10))
# seed(1)
# print(randrange(10))
# import pandas as pd
# data = pd.read_csv('sonar.csv')
# print(data)
import numpy as np
from sklearn.model_selection import cross_val_predict
#
# listA = [i for i in range(100)]
# print(listA)
# cross_val_predict()
from sklearn.model_selection import train_test_split

from Assignment2.bagging.classifierTest import naiveClassifier

listA = [[(j + 10 * i) for j in range(10)] for i in range(10)]
data = np.array(listA)
dataX = data[:, :-1]
dataY = data[:, -1]
model = naiveClassifier()
prediction = cross_val_predict(model, dataX, dataY, cv=10)
print(prediction)



import numpy as np
import pandas as pd

# a = [[1, 2, 3, 4]]
# b = [5]
# aa = np.array(a)
# bb = np.array(b)
# cc = np.c_[aa, bb]
# print(cc.shape)
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict

from Assignment2.bagging.classifierTest import naiveClassifier

# model = naiveClassifier()
# test = np.empty((1, 4))
# testY = model.predict(test)
# print(testY[0])
# predictions = [1, 2, 3, 4, 5, 6, 1]
# most = max(set(predictions), key=predictions.count)
# print(most)
# a = np.array([1, 2, 3, 4])
# a = a.reshape(1, -1)
# print(a)
data = pd.read_csv('divorce.csv', delimiter=';')
data = data.values
dataX = data[:, : -1]
dataY = data[:, -1]
model = BaggingClassifier()
prediction = cross_val_predict(model, dataX, dataY, cv=10)
print(accuracy_score(dataY, prediction))
print(precision_score(prediction, dataY))
print(recall_score(prediction, dataY))
print(f1_score(prediction, dataY))

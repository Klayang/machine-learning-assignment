import pandas as pd

from Assignment2.adaboost.AdaBoostClassifier import AdaBoostClassifier
from Assignment2.adaboost.adaboost_scratch import AdaBoost_scratch
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from Assignment2.adaboost.cross_val_predict import cross_val_predict

data = pd.read_csv('divorce.csv', delimiter=';')
data = data.values
dataX = data[:, : -1]
dataY = data[:, -1]
prediction = cross_val_predict(data, 'NB', 50)
print(accuracy_score(dataY, prediction))
print(precision_score(prediction, dataY))
print(recall_score(prediction, dataY))
print(f1_score(prediction, dataY))

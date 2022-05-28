import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import BernoulliNB
import pandas as pd
from sklearn.svm import SVC

data = pd.read_csv('divorce.csv', delimiter=';')
data = data.values
dataX = data[:, : -1]
dataY = data[:, -1]
model = AdaBoostClassifier(BernoulliNB())
prediction = cross_val_predict(model, dataX, dataY, cv=10)
print(accuracy_score(dataY, prediction))
print(precision_score(prediction, dataY))
print(recall_score(prediction, dataY))
print(f1_score(prediction, dataY))

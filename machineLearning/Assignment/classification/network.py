import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

data = pd.read_csv('divorce.csv', delimiter=';')
features = data.columns.tolist()
features.remove("Class")
features = features[0:2]
target = 'Class'


trainX, testX, trainY, testY = train_test_split(data[features], data[target], test_size=0.4, random_state=32)
trainX = trainX.values
trainY = trainY.values
testX = testX.values
testY = testY.values

model = MLPClassifier(solver='sgd', learning_rate='constant', momentum=0, learning_rate_init=0.1, max_iter=500)
model.fit(trainX, trainY)
prediction = model.predict(testX)
acc = accuracy_score(prediction, testY)
pre = precision_score(prediction, testY)
rec = recall_score(prediction, testY)
f1 = f1_score(prediction, testY)
print(acc)
print(pre)
print(rec)
print(f1)

ite_list = []
acc_list = []
for i in range(10, 500, 10):
    ite_list.append(i)
    model = MLPClassifier(solver='sgd', learning_rate='constant', momentum=0, learning_rate_init=0.1, max_iter=i)
    model.fit(trainX, trainY)
    prediction = model.predict(testX)
    acc = accuracy_score(prediction, testY)
    acc_list.append(acc)
print(ite_list)
print(acc_list)
plt.plot(ite_list, acc_list, '-')
plt.xlabel('iterations')
plt.ylabel('accuracy')
plt.show()

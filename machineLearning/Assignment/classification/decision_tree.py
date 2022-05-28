import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('divorce.csv', delimiter=';')
features = ['Atr1', 'Atr2', 'Atr3']
target = ['Class']
data = data[features + target]

trainX, testX, trainY, testY = train_test_split(data[features], data[target], test_size=0.4, random_state=32)
trainX = trainX.values
trainY = trainY.values
testX = testX.values
testY = testY.values

model = DecisionTreeClassifier(max_depth=20)
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

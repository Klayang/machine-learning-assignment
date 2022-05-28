import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('divorce.csv', delimiter=";")
print(type(data))

features = ['Atr1', 'Atr2', 'Atr3']
target = 'Class'
data = data[features + [target]]
print(data.head())

trainX, testX, trainY, testY = train_test_split(data[features], data[target], test_size=0.4, random_state=32)
trainX = trainX.values
trainY = trainY.values
testX = testX.values
testY = testY.values
print(trainX)

model = LogisticRegression()
model.fit(trainX, trainY)
prediction = model.predict(testX)
print(prediction.shape)
acc = accuracy_score(prediction, testY)
pre = precision_score(prediction, testY)
rec = recall_score(prediction, testY)
f1 = f1_score(prediction, testY)
print(acc)
print(pre)
print(rec)
print(f1)

#本题中我们采用1-10个属性,展示使用不同数量和种类的属性时模型的泛化效果
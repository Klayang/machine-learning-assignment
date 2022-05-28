import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

data = pd.read_excel('Folds.xlsx')

features = ['AT', 'V', 'AP', 'RH']
target = ['PE']
data = data[features + target]

trainX, testX, trainY, testY = train_test_split(data[features], data[target], test_size=0.4, random_state=32)
trainX = trainX.values
trainY = trainY.values
testX = testX.values
testY = testY.values

model = MLPRegressor(solver='sgd', learning_rate='constant', momentum=0, learning_rate_init=0.1, max_iter=500)
model.fit(trainX, trainY)
prediction = model.predict(testX)
mae = mean_absolute_error(prediction, testY)
mse = mean_squared_error(prediction, testY)
rmse = mse ** 0.5
print(mae)
print(mse)
print(rmse)
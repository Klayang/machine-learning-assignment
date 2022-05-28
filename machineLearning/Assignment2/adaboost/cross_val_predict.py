import numpy as np

from Assignment2.adaboost import AdaBoostClassifier


def cross_validation_split(dataset):
    singleSize = len(dataset) / 10
    singleSize = int(singleSize)
    data_split = []
    for i in range(10):
        data_split.append(dataset[singleSize * i: singleSize * (i + 1)])
    return data_split


def cross_val_predict(data, modelName='decision_tree', num=10):
    folds = cross_validation_split(data)
    predictions = []
    for i in range(10):
        trainSet = list(folds)
        testData = trainSet.pop(i)
        testX = testData[:, : -1]
        trainData = trainSet[0]
        for j in range(1, 9):
            trainData = np.vstack((trainData, trainSet[j]))
        trainX = trainData[:, :-1]
        trainY = trainData[:, -1]

        model = AdaBoostClassifier(num, modelName)
        model.fit(trainX, trainY)
        prediction = model.predict(testX)
        predictions.append(prediction)

    predictions = np.array(predictions).reshape(-1)
    return predictions

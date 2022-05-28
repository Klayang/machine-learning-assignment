from random import randrange

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from Assignment2.bagging.cross import cross_val_predict


def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


data = pd.read_csv('divorce.csv', delimiter=';')
data = data.values
dataX = data[:, : -1]
dataY = data[:, -1]
predictions = cross_val_predict(data, 'NB', 20)
print(accuracy_score(predictions, dataY))
print(precision_score(predictions, dataY))
print(recall_score(predictions, dataY))
print(f1_score(predictions, dataY))


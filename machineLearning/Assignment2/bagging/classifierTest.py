import numpy as np
class naiveClassifier:
    def __init__(self, l=0.01):
        self.l = l

    def fit(self, trainX, trainY):
        i = 1

    def predict(self, testX):
        length = len(testX)
        return np.zeros(length)

    def get_params(self, deep=False):
        return {'l': self.l}

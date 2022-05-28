import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from Assignment2.bagging import subsample


class baggingClassifier:
    def __init__(self, num=10, model='decision_tree', l=0.01):
        try:
            self.l = l
            self.modelDict = {}
            self.nums = num
            self.modelList = ['decision_tree', 'SVM', 'LR', 'neural_network', 'NB']
            if model not in self.modelList:
                raise Exception('不存在您要求的分类器')
            self.modelDict['decision_tree'] = DecisionTreeClassifier(max_depth=5)
            self.modelDict['SVM'] = SVC(C=1, kernel='rbf')
            self.modelDict['LR'] = LogisticRegression()
            self.modelDict['neural_network'] = MLPClassifier(solver='sgd', learning_rate='constant', momentum=0,
                                                             learning_rate_init=0.1, max_iter=500)
            self.modelDict['NB'] = BernoulliNB()
            self.model = self.modelDict[model]
            self.models = []
        except Exception as e:
            print(e)

    def fit(self, trainX, trainY):
        trainData = np.c_[trainX, trainY]
        for i in range(self.nums):
            sample = subsample(trainData, 0.50)
            model = self.model
            sampleX = sample[:, :-1]
            sampleY = sample[:, -1]
            model.fit(sampleX, sampleY)
            self.models.append(model)

    def __bagging_predict(self, row):
        row = row.reshape(1, -1)
        predictions = [model.predict(row)[0] for model in self.models]
        # predictions = []
        # for model in self.models:
        #     value = model.predict(row)
        #     sign = value[0]
        #     predictions.append(sign)
        return max(set(predictions), key=predictions.count)

    def predict(self, testX):
        predictions = np.array([self.__bagging_predict(row) for row in testX])
        print(predictions.shape)
        return predictions

    def get_params(self, deep=False):
        return {'l': self.l}

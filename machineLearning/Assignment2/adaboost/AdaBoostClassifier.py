import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class AdaBoostClassifier:
    def __init__(self, num=10, model='decision_tree', learning_rate = 1):
        try:
            self.modelDict = {}
            self.nums = num
            self.modelList = ['decision_tree', 'SVM', 'LR', 'neural_network', 'NB']
            if model not in self.modelList:
                raise Exception('不存在您要求的分类器')
            self.modelDict['decision_tree'] = DecisionTreeClassifier(max_depth=2)
            self.modelDict['SVM'] = SVC(C=0.1, kernel='rbf')
            self.modelDict['LR'] = LogisticRegression()
            self.modelDict['neural_network'] = MLPClassifier(solver='sgd', learning_rate='constant', momentum=0,
                                                             learning_rate_init=0.1, max_iter=500)
            self.modelDict['NB'] = BernoulliNB()
            self.model = self.modelDict[model]
            self.learning_rate = learning_rate
            self.estimator_list = []
            self.y_predict_list = []
            self.estimator_error_list = []
            self.estimator_weight_list = []
            self.sample_weight_list = []

        except Exception as e:
            print(e)

    def fit(self, trainX, trainY):
        N = len(trainY)
        sample_weight = np.ones(N) / N  # 所有样本权重均为1 / N
        self.sample_weight_list.append(sample_weight.copy())

        # For m = 1 to M
        for m in range(self.nums):
            # Fit a classifier
            estimator = self.model
            estimator.fit(trainX, trainY, sample_weight=sample_weight)
            y_predict = estimator.predict(trainX)

            # Misclassifications
            incorrect = (y_predict != trainY)

            # Estimator error
            estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

            # Boost estimator weights
            estimator_weight = self.learning_rate * np.log((1. - estimator_error) / (estimator_error + 1e-6))

            # Boost sample weights
            sample_weight *= np.exp(estimator_weight * incorrect * ((sample_weight > 0) | (estimator_weight < 0)))

            # Save iteration values
            self.estimator_list.append(estimator)
            self.estimator_error_list.append(estimator_error.copy())
            self.estimator_weight_list.append(estimator_weight.copy())
            self.sample_weight_list.append(sample_weight.copy())

    def predict(self, testX):
        for model in self.estimator_list:
            self.y_predict_list.append(model.predict(testX))
        self.y_predict_list = np.array(self.y_predict_list)
        preds = (np.array([np.sign((self.y_predict_list[:, point] * self.estimator_weight_list).sum()) for point in range(len(testX))]))
        return preds

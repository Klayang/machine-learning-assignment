import numpy as np
from sklearn.tree import DecisionTreeClassifier


def AdaBoost_scratch(X, y, M=10, learning_rate=1):
    # Initialization of utility variables
    N = len(y)
    # 定义5个列表：分类器，分类结果，错误样本，分类器权重列表，样本权重列表
    estimator_list, y_predict_list, estimator_error_list, estimator_weight_list, sample_weight_list = [], [], [], [], []

    # Initialize the sample weights
    sample_weight = np.ones(N) / N  # 所有样本权重均为1 / N
    sample_weight_list.append(sample_weight.copy())

    # For m = 1 to M
    for m in range(M):
        # Fit a classifier
        estimator = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
        estimator.fit(X, y, sample_weight=sample_weight)
        y_predict = estimator.predict(X)

        # Misclassifications
        incorrect = (y_predict != y)

        # Estimator error
        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

        # Boost estimator weights
        estimator_weight = learning_rate * np.log((1. - estimator_error) / estimator_error)

        # Boost sample weights
        sample_weight *= np.exp(estimator_weight * incorrect * ((sample_weight > 0) | (estimator_weight < 0)))

        # Save iteration values
        estimator_list.append(estimator)
        y_predict_list.append(y_predict.copy())
        estimator_error_list.append(estimator_error.copy())
        estimator_weight_list.append(estimator_weight.copy())
        sample_weight_list.append(sample_weight.copy())

    # Convert to np array for convenience
    estimator_list = np.asarray(estimator_list)
    y_predict_list = np.asarray(y_predict_list)
    estimator_error_list = np.asarray(estimator_error_list)
    estimator_weight_list = np.asarray(estimator_weight_list)
    sample_weight_list = np.asarray(sample_weight_list)

    # Predictions
    preds = (np.array([np.sign((y_predict_list[:, point] * estimator_weight_list).sum()) for point in range(N)]))
    print('Accuracy = ', (preds == y).sum() / N)

    return estimator_list, estimator_weight_list, sample_weight_list
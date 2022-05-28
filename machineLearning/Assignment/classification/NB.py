import numpy as np
import math
class myGaussianNB:
    '''
    处理连续特征的高斯朴素贝叶斯
    '''

    def __init__(self):
        '''
        初始化四个字典
        self.label_mapping     类标记 与 下标(int)
        self.probability_of_y  类标记 与 先验概率(float)
        self.features          属性标记 与 类别数
        self.conditional_probability 类标记 与 条件概率(float)
        self.samples           类标记 与 样本数
        self.attributes        类别属性 与 具体属性
        '''
        self.label_mapping = dict()
        self.probability_of_y = dict()
        self.features = dict()
        self.samples = dict()
        self.conditional_probability = dict()
        self.attributes = dict()

    def _clear(self):
        '''
        为了防止一个实例反复的调用fit方法，我们需要每次调用fit前，将之前学习到的参数删除掉
        '''
        self.label_mapping.clear()
        self.probability_of_y.clear()
        self.features.clear()
        self.conditional_probability.clear()
        self.samples.clear()

    def fit(self, trainX, trainY):
        '''
        这里，我们要根据trainY内的类标记，针对每类，计算这类的先验概率，以及这类训练样本每个特征的均值和方差

        Parameters
        ----------
            trainX: np.ndarray, 训练样本的特征, 维度：(样本数, 特征数)

            trainY: np.ndarray, 训练样本的标记, 维度：(样本数, )
        '''

        # 先调用_clear
        self._clear()

        # 获取类标记
        labels = np.unique(trainY)

        # 添加类标记与下标的映射关系
        self.label_mapping = {label: index for index, label in enumerate(labels)}

        # 获取每一列属性对应的类别数
        for i in range(0, trainX.shape[1]):
            self.features[i] = len(np.unique(trainX[:, i]))

        # 遍历每个类
        for label in labels:
            # 取出为label这类的所有训练样本，存为 x
            x = trainX[trainY == label, :]

            # 计算每一个类所对应的样本个数
            self.samples[label] = len(x)

            # 计算先验概率，用 x 的样本个数除以训练样本总个数，存储到 self.probability_of_y 中，键为 label，值为先验概率
            # YOUR CODE HERE
            self.probability_of_y[label] = (len(x) + 1) / (len(trainX) + len(labels))

            # 计算条件概率
            features_list = []
            len_x = self.samples[label]

            attributes_with_this_label = []

            for i in range(trainX.shape[1]):
                features_type = {}
                features = np.unique(x[:, i])
                attributes_with_this_label.append(features)
                for j in range(len(features)):
                    features_type[features[j]] = (len(x[x[:, i] == features[j]]) + 1) / (len_x + len(features))
                features_list.append(features_type)

            self.conditional_probability[label] = features_list
            self.attributes[label] = attributes_with_this_label

    def predict(self, testX):
        '''
        给定测试样本，预测测试样本的类标记，这里我们要实现化简后的公式

        Parameters
        ----------
            testX: np.ndarray, 测试的特征, 维度：(测试样本数, 特征数)

        Returns
        ----------
            prediction: np.ndarray, 预测结果, 维度：(测试样本数, )
        '''

        # 初始化一个空矩阵 results，存储每个样本属于每个类的概率，维度是 (测试样本数，类别数)，每行表示一个样本，每列表示一个特征
        results = np.empty((testX.shape[0], len(self.probability_of_y)))

        # 初始化一个列表 labels，按 self.label_mapping 的映射关系存储所有的标记，一会儿会在下面的循环内部完成存储
        labels = [0] * len(self.probability_of_y)

        # 遍历当前的类，label为类标记，index为下标，我们将每个样本预测出来的这个 label 的概率，存到 results 中的第 index 列
        for label, index in self.label_mapping.items():

            # 先验概率存为 py
            py = self.probability_of_y[label]

            # 取出存放对应label的条件概率的list
            feature_list = self.conditional_probability[label]

            # 取出存放对应label的属性列表attribute_list
            attribute_list = self.attributes[label]

            result = np.zeros(len(testX))

            for i in range(len(testX)):
                for j in range(testX.shape[1]):
                    if testX[i][j] in attribute_list[j]:
                        result[i] += np.log(feature_list[j][testX[i, j]])
                    else:
                        result[i] += (1 / (self.samples[label] + self.features[j]))

            result += np.log(py)

            # 将当前的label，按index顺序放入到labels中
            labels[index] = label

            # debug
            assert result.shape == (len(testX),)

            # 将所有测试样本属于当前这类的概率，存入到results中
            results[:, index] = result

            # 将当前的label，按index顺序放入到labels中
            labels[index] = label

        # 将labels转换为np.ndarray
        np_labels = np.array(labels)

        # 循环结束后，就计算出了给定测试样本，当前样本属于这类的概率的近似值，存放在了results中，每行对应一个样本，每列对应一个特征
        # 我们要求每行的最大值对应的下标，也就是求每个样本，概率值最大的那个下标是什么，结果存入max_prob_index中
        # YOUR CODE HERE
        max_prob_index = np.argmax(results, axis = 1)

        # debug
        assert max_prob_index.shape == (len(testX),)

        # 现在得到了每个样本最大概率对应的下标，我们需要把这个下标变成 np_labels 中的标记
        # 使用上面小技巧中的第五点求解
        # YOUR CODE HERE
        prediction = np_labels[max_prob_index]

        # debug
        assert prediction.shape == (len(testX),)

        # 返回预测结果
        return prediction
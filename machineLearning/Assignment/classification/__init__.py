import numpy as np
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from Assignment.classification.NB import myGaussianNB
from sklearn.model_selection import train_test_split

spambase = np.loadtxt('spambase.data', delimiter = ",")
spamx = spambase[:, :57]
spamy = spambase[:, 57]
spamx_binary = (spamx != 0).astype('float64')
trainX, testX, trainY, testY = train_test_split(spamx_binary, spamy, test_size = 0.4, random_state = 32)
print(testX[0])

model = myGaussianNB()
model.fit(trainX, trainY)
prediction = model.predict(testX)
print(accuracy_score(testY, prediction))

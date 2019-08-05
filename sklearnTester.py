# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

data = sio.loadmat('data/trump.mat')
histograms = data['histograms']

codebook_size = histograms.shape[1]


#histograms = histograms + 1;
labels = data['labels']

X_train, X_test, y_train, y_test = train_test_split(histograms, labels, test_size=0.2)

#clf = MultinomialNB()
y_train = y_train.ravel()
#clf.fit(X_train, y_train)

clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)

correct = 0;
predict = np.zeros((len(y_test),1))

for i in range(len(y_test)):
    vec = X_test[i].reshape(1,codebook_size)
    predict[i] = clf.predict(vec)
    if predict[i] == y_test[i]:
        correct = correct + 1;
        
acc = correct/len(y_test);

cm = confusion_matrix(y_test, predict)


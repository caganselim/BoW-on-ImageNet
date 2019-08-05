# -*- coding: utf-8 -*-
import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from plot_cm import plot_confusion_matrix
import time


data = sio.loadmat('data/k2000.mat')
histograms = data['histograms']

codebook_size = histograms.shape[1]

#histograms = histograms + 1;
labels = data['labels']

X_train, X_test, y_train, y_test = train_test_split(histograms, labels, test_size=0.2,random_state=2)

start_time = time.time()
#Train
train_size = X_train.shape[0]
codebook_size = X_train.shape[1]
noOfClasses = 10

noOfOccurences = np.ones((noOfClasses,codebook_size))
prior = np.zeros((noOfClasses,1))

for i in range(0,train_size):
    
    classId = y_train[i] - 1
    noOfOccurences[classId] = noOfOccurences[classId] + X_train[i]
    prior[classId] = prior[classId] + 1
    
elapsed = time.time() - start_time

classWordCounts = np.sum(noOfOccurences, axis=1).reshape((noOfClasses,1))

likelihood = np.zeros((noOfClasses,codebook_size))
likelihood = np.log(np.divide(noOfOccurences,classWordCounts))

prior = prior/train_size

#End of train

#Test
out = np.matmul(X_test, np.transpose(likelihood)) + np.transpose(np.log(prior))

#Get predictions
predict = np.argmax(out,axis=1)

#Calculate confusion matrix
cm = confusion_matrix(y_test-1, predict)

#Find accuracy
acc = np.sum(np.diag(cm))/np.sum(np.sum(cm))

plot_confusion_matrix(cm,'Confusion Matrix for Naive Bayes')
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import scipy.io as sio
from plot_cm import plot_confusion_matrix
import matplotlib.pyplot as plt
import time

# M TRAIN IMAGES
#4848-M TEST IMAGES
#64 FEATURES PER IMAGE
#10 class

#x -- inputs(MX64)
#W -- weights(64x10)
#B -- bias(Mx10)
#h -- induced local field(Mx10)
#d -- expected(Mx10)

class LogisticRegression:
    def __init__(self,lr,w,d,cl):
        self.lr = lr;
        self.w =w
        self.d = d
        self.cl = cl
    def __softmax(self,x):
        sft = np.zeros((x.shape[0],self.cl))
        for i in range(x.shape[0]):
            z = x[i,:]        
            sft[i,:] = np.exp(z)/ np.sum([np.exp(c) for c in z])
           
        return sft
    
    def __loss(self,h,d):

        lossy = -np.sum(d*np.log(h+1e-6))/d.shape[0]
        return lossy

    
    def __fit(self,x,d):
        biasInput = np.ones((x.shape[0],1))
        x = np.concatenate((biasInput,x), axis=1)
        v = np.dot(x,self.w)
        phi = self.__softmax(v)
        gradient = np.dot(x.T,(phi-d))/d.shape[0]
        self.w = self.w - self.lr*gradient
        
        return self.__loss(phi,d)
     
        
        
    def train(self, x, d, batch_size, iteration, y_test , X_test):
        
        s_time = time.time()
        lossContainer = np.zeros((iteration,1))
        accContainer = np.zeros((iteration,1))
        testContainer = np.zeros((iteration,1))
        for itr in range(iteration):
            
            cumulative_loss = 0
            
            print('Train at iteration ' + str(itr+1) + '/' + str(iteration) + ' with batch size ' + str(batch_size))
        
            train_size = x.shape[0]
            
            indexes = np.arange(train_size)
            np.random.shuffle(indexes)
            
            full_batch_size = int(train_size/batch_size)
            
            for i in range(full_batch_size):
                                
                start_idx = i*batch_size
                finish_idx = (i+1)*batch_size - 1
                
                idx = indexes[start_idx:finish_idx]
                
                x_batch = x[idx,:]
                d_batch = d[idx,:]
                
                loss =  self.__fit(x_batch,d_batch)
                
                cumulative_loss = cumulative_loss + loss
            
            
            if train_size%batch_size != 0:
                partial_start_idx = full_batch_size*batch_size
                partial_finish_idx = train_size - 1
                
                idx = indexes[partial_start_idx:partial_finish_idx]
                    
                x_batch = x[idx,:]
                d_batch = d[idx,:]
                           
                loss = self.__fit(self,x_batch,d_batch)
                
                cumulative_loss = cumulative_loss + loss
                
            lossContainer[itr] = cumulative_loss
            print('Loss: ' + str(cumulative_loss) + ' at iteration ' + str(itr+1) + '/' + str(iteration))
            
            #Train Predict
            train_pred = self.predict(x)
                
            #Confusion Matrix for Test
            cm = confusion_matrix(np.argmax(d,axis=1), train_pred)
    
            #Accuracy
            acc = np.sum(np.diag(cm))/np.sum(np.sum(cm))
            accContainer[itr] = acc
            print('Train accuracy: ' + str(acc) + ' at iteration ' + str(itr+1) + '/' + str(iteration))
            
            #Test Predict
            test_pred = self.predict(X_test)
    
            #Confusion Matrix for Test
            cm = confusion_matrix(y_test, test_pred)
    
            #Accuracy
            acc = np.sum(np.diag(cm))/np.sum(np.sum(cm))
            testContainer[itr] = acc            
        
        elapsed = time.time() - s_time
        print('Elapsed time: ' + str(elapsed))
        #Plot
        plt.plot(lossContainer)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Cross Entropy Loss vs. Iterations - Batch Size:' + str(batch_size))
        plt.show()
        plt.close()
        
        plt.plot(accContainer)
        plt.plot(testContainer)
        plt.legend(('Train', 'Test'))
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. Iterations - Batch Size:' + str(batch_size))
        plt.show()
        plt.close()
            
            
    def predict(self,x):
        
        biasInput = np.ones((x.shape[0],1))
        x = np.concatenate((biasInput,x), axis=1)
        
        v = np.dot(x,self.w)
        prob =  self.__softmax(v)
        return np.argmax(prob,axis=1)
    
        

if __name__== "__main__":
    
    #Load Data
    data = sio.loadmat('data/k2000.mat')
    histograms = data['histograms']
    codebook_size = histograms.shape[1]
    
    #Quick fix for class indexes (MATLAB vs. Python)
    labels = data['labels'] - 1
    
    histograms = histograms.astype('float32')
    norms = np.linalg.norm(histograms, axis=1) + 0.0001
    
    normalized_histograms = np.divide(histograms,norms.reshape(3500,1))
    normalized_histograms = np.divide((normalized_histograms - np.mean(normalized_histograms, axis=0)), np.std(normalized_histograms, axis=0))
    

    #Split the dataset    
    X_train, X_test, y_train, y_test = train_test_split(normalized_histograms, labels, test_size=0.4, random_state = 1)
    y_train = y_train.ravel()

    
    M = X_train.shape[0] #Number of train images
    f = X_train.shape[1] #Number of features
    lr= 0.01 #learning rate
    x = X_train #Input
    cl = 10 #Class size
    w =  np.random.uniform(-0.05, 0.05, (f+1,cl)) #Weights
    od =  y_train #Labels
    
    #Train Parameters
    batch_size = 75
    iteration_size = 50
    
    #one hot encoding for labels
    d = np.zeros((M,cl))
    for i in range(M):
        print(i)
        d[i,od[i]]=1
        
    #Construct regression object
    model = LogisticRegression(lr,w,d,cl)
    
    #Train the model
    model.train(x,d, batch_size, iteration_size, y_test, X_test)
    
    #Predict
    test_pred = model.predict(X_test)
    
    #Confusion Matrix for Test
    cm = confusion_matrix(y_test, test_pred)
    
    #Accuracy
    acc = np.sum(np.diag(cm))/np.sum(np.sum(cm))
    
    #Plot confusion matrix
    plot_confusion_matrix(cm,'Confusion Matrix for Logistic Regression - K = 2000, Batch Size = 75, Iteration = 50')
# -*- coding: utf-8 -*-
import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def oneHOTEncoder(labels, classSize):
    
    setSize = labels.shape[0]
    oneHotLabels = np.zeros((labels.shape[0] , classSize))
    
    for k in range(0, setSize):
        val = labels[k]
        oneHotLabels[k,val-1] = 1;        
    
    
    return oneHotLabels


class FCLayer():
    def __init__(self, inputSize, neuronSize):
        
        self.weightMatrix = np.random.uniform(-0.02, 0.02, (inputSize+1,neuronSize))
    
    #Useful print function
    def __repr__(self):
        
       return str(self.weightMatrix.shape)
        
class FCNetwork():
    def __init__(self,dimensionVector):
    
        #Dimensions accept
        self.noOfLayers = layerSizes.size
        self.layers = [FCLayer(layerSizes[0],layerSizes[1])]
        
        #Concat additional layers
        for i in range(1,self.noOfLayers-1):
            additionalLayer = FCLayer(layerSizes[i], layerSizes[i+1])
            self.layers.append(additionalLayer)
        
        print('Constructed FCNetwork with weights: ' + str(self.layers))
                    
        
    def __sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def __der_sigmoid(self,x):
        
        return self.__sigmoid(x)*(1-self.__sigmoid(x))
        
    
    def forward(self, inputVector):
        
        prevLayerInputs = list() #Container for inputs
        
        codebook_size = inputVector.shape[0]
        inputVector = inputVector.reshape((codebook_size, 1))
          
        #Save
        inputVector = np.concatenate((inputVector, np.ones((1,1))))
      
        prevLayerInputs.append(inputVector)
        
        #Output
        firstOutput = np.matmul(np.transpose(self.layers[0].weightMatrix), inputVector)
        firstOutput = np.concatenate((firstOutput, np.ones((1,1))))
        prevLayerInputs.append(firstOutput)
        
        for i in range(1, self.noOfLayers - 1):
            
            #Prev output
            prevOut = self.__sigmoid(prevLayerInputs[i])
        
            #Add bias term
            currOut = np.matmul(np.transpose(self.layers[i].weightMatrix), prevOut)
                       
            if i != self.noOfLayers - 2:
                currOut = np.concatenate((currOut, np.ones((1,1))))
                prevLayerInputs.append(currOut)
            else:
                output = currOut
                    
        return  output, prevLayerInputs
 
    def backward(self,prevLayerInputs,error):
        
        weightUpdates = list()
        gradients = list()
        
        gradients.append(error)
        dw_last = np.matmul(prevLayerInputs[self.noOfLayers - 2], np.transpose(error))
                
        weightUpdates.append(dw_last)
           
        #Ok great, moving on
        for i in range(1, self.noOfLayers - 1):
            
            grad_out = gradients[i-1] #OK
            #print('grad_out {}'.format(grad_out.shape))
            
            #Get outer weights
            w_out = self.layers[self.noOfLayers-i-1].weightMatrix
            #print('w_out  {}'.format(w_out.shape))
            

            #Propagate error
            outerPunishment = np.matmul(w_out,grad_out)
            #print('outerPunishment {}'.format(outerPunishment.shape))
                      
            #Input
            inputEnteringThisLayer = prevLayerInputs[self.noOfLayers-i-2]
            #print('inputEnteringThisLayer {}'.format(inputEnteringThisLayer.shape))
            
            
            #Output
            outputFromThisLayer = prevLayerInputs[self.noOfLayers-i-1]
            
            #Current layer       
            der_out = self.__der_sigmoid(outputFromThisLayer) #OK
            #print('der_out {}'.format(der_out.shape))
                                  
            #Get current local gradient           
            currentLocalGradient = np.multiply(der_out,outerPunishment)
            #print('currentLocalGradient {}'.format(currentLocalGradient.shape))
            
            #Append local gradient
            
            currentLocalGradient = currentLocalGradient[0:currentLocalGradient.shape[0] - 1,:]
            gradients.append(currentLocalGradient) #1x50, 50x1
            
            #Current weight
            deltaCurrLayer = np.matmul(inputEnteringThisLayer,currentLocalGradient.T)
            
            #Save
            weightUpdates.append(deltaCurrLayer)
            
            #print('OK')
            
        
        return weightUpdates
        
    def train(self, data, labels, lr, batch_size, epoch,X_test,y_test):   
        labels = oneHOTEncoder(labels, 10)
        y_test = oneHOTEncoder(y_test,10) 
       
        weightContainer = list()
        
        for i in range( len(self.layers) ):
            
            weightContainer.append(np.zeros(self.layers[i].weightMatrix.shape))
            
        mse = np.zeros(epoch)
        acc = np.zeros(epoch)
        test_acc = np.zeros(epoch)
        test_mse = np.zeros(epoch)
        
        for e in range(epoch):
            #One epoch
           for i in range(data.shape[0]):
           
                #Perform forward prop

                
                output, prevLayerInputs = self.forward(data[i])
                
                #print(output)
                
                desired = labels[i].reshape((10,1))
                error = desired - output #Sum gradients
                
                #Perform backprop
                weightUpdates = self.backward(prevLayerInputs, error)
                
                
                
                weightUpdates.reverse()
                
                #print(output)
                                
                #print(i)
                if i % batch_size == 0 and i!=0:
                    
                    #print('Update at batch {}, at epoch {} '.format(i, e+1))
                
                    #Weight update
                    for k in range( self.noOfLayers - 1 ):
                       
                       #print( lr*(weightContainer[k]/batch_size))
                       self.layers[k].weightMatrix += lr*(weightContainer[k]/batch_size)
                       
                    #Reset container
                        
                    for k in range(self.noOfLayers - 1 ):
                       weightContainer[k] *= 0.9
                       #weightContainer.append(np.zeros(self.layers[k].weightMatrix.shape))
                    
                else:
                    #Sum
                    for k in range( self.noOfLayers - 1 ):
                        
                        weightContainer[k] += weightUpdates[k]
                
           #MSE
           train_predictions, train_mse, train_acc = self.test(data, labels)
           print('Train MSE= {}, ACC= {} at epoch {} '.format(train_mse, train_acc,  e+1))
           
           acc[e] = train_acc
           mse[e] = train_mse
           
           
           predictions, mse2, acc2 = neuralNet.test(X_test,y_test)
              
           test_acc[e] = acc2
           test_mse[e] = mse2   
                
        return acc, mse, test_acc, test_mse
                        
        
    def test(self, data, labels):
        
        test_size = data.shape[0];
        predictions = np.zeros(labels.shape)
        
        correct = 0
        mse = 0
        
                
        for i in range(0,test_size):
            
            
            output, prevLayerInputs = self.forward(data[i])
            
            #print(np.square(labels[i] - output.flatten()))
            
            
            mse += np.sum(np.square(labels[i] - output.flatten()))            
            
            classId = np.argmax(output)
            #print(classId)
            predictions[i] = classId 

            #print(np.argmax(labels))
            if classId == np.argmax(labels[i]):
                
                correct += 1

        
        #print('Test')
        mse =  0.5*mse/test_size
        acc = correct/test_size
        
        return predictions, mse, acc



    
    def __repr__(self):
        
        return 'Neural Net';
        
    
if __name__== "__main__":
    
    
    data = sio.loadmat('data/k500.mat')
    histograms = data['histograms']

    codebook_size = histograms.shape[1]
    
    #histograms = histograms + 1;
    labels = data['labels']
    
    
    histograms = histograms.astype('float32')
    norms = np.linalg.norm(histograms, axis=1) + 0.0001
    normalized_histograms = np.divide(histograms,norms.reshape(3500,1))
    normalized_histograms = np.divide((normalized_histograms - np.mean(normalized_histograms, axis=0)), np.std(normalized_histograms, axis=0))

    X_train, X_test, y_train, y_test = train_test_split(normalized_histograms, labels, test_size=0.2)
    
    class_size = 10;
    
    #Define the Neural Network
    layerSizes = np.array([codebook_size,200,50,class_size])
    
    
    #Create the NN object
    neuralNet = FCNetwork(layerSizes)
    
    lr = 0.05
    batch_size = 100
    epoch = 20
    
    acc, mse, test_acc, test_mse = neuralNet.train(X_train, y_train, lr, batch_size,epoch,X_test,y_test)
    
#    vvec = oneHOTEncoder(y_test,10)
#
#    predictions, mse, acc = neuralNet.test(X_test, vvec)
#    
#    p = predictions[:,0] + 1
#        
#    from plot_cm import plot_confusion_matrix
#    cm = confusion_matrix(y_test , p)
#    plot_confusion_matrix(cm,'Confusion Matrix for Neural Network')

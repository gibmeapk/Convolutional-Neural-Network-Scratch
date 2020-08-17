from layers.layer import *
from layers.utility.initializer import *

import numpy as np

class fullyConnected(layer):
    def __init__(self,out,learningRate):
        
        '''
        Initalizes hyperparameters
        '''
        
        self.inn = 0
        self.out = out
        self.LEARNING_RATE = learningRate

        self.cache = []
        self.bias = []
        self.weights = []
        
    def initalizeWeight(self, channel):
        
        '''
        calls He normal initalizer (found in utility.py) to define inital weight
        '''
            
        self.weights =  He_initialization(channel,self.out)
        
    def initalizeBias(self):
        
        '''
        sets inital bias
        '''
        
        self.bias =  np.zeros((self.out,1))
        
    def forward(self,image):

        '''
        Performs forward propergation
        '''
        
        self.cache = image
        
        
        if len(self.weights) == 0:
            channel = self.cache.shape[1]
            self.initalizeWeight(channel)

        if len(self.bias) == 0:
            self.initalizeBias()
            
        dp = self.cache @ self.weights
        output = dp + self.bias.transpose()
        return output

    def backward(self,backImage):
        '''
        Performs back propergation of fully-connected layer
        '''
        backImage = backImage.transpose()
        pastDeriv = backImage.transpose() @ self.weights.transpose()
        weightDeriv = (backImage.dot(self.cache)).transpose()
        biasDeriv = np.sum(backImage, axis=1, keepdims=True)
        self.updateParams(weightDeriv,biasDeriv)
        
        return pastDeriv

    def updateParams(self,w,b):
        '''
        Updates hyperparameters weight and bias
        '''
        self.weights = self.weights - (self.LEARNING_RATE * w)
        self.bias = self.bias - (self.LEARNING_RATE * b)

        

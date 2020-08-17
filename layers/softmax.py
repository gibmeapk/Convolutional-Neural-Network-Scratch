from layers.layer import *

import numpy as np
import warnings
warnings.filterwarnings("error") #This prevents the overflow error from appearing on stdout


class softmax(layer):
    
    '''
    Calculates the softmax output of the network
    the output presents the probabilities for each class
    '''
    
    def __init__(self):
        self.cache = []
        
    def forward(self, image):
        
        '''
        Performs forward propergation of softmax
        '''
        
        try:
            exp = np.exp(image,dtype=np.float64)
            self.cache = exp/np.sum(exp)
            return self.cache

        except RuntimeWarning:
            self.forwardStopOverflow(image)

    def forwardStopOverflow(self,image):
        
        '''
        np.exp is prone to overflow errors, as a result this method is present to calculate
        np.exp IF overflow occurs
        '''
        print("OVERFLOW HAS OCCURED")
        
        maxImg = image.max()
        exp = np.exp(image - maxImg)
        self.cache = exp / exp.sum()
        return self.cache
    
    def backward(self, x):
        
        '''
        Performs backwards propergation of softmax
        '''
        
        cache = self.cache.transpose()
        x = x.reshape(x.shape[0],1)
        output = cache - x
        return output.transpose()

import numpy as np
from layers.layer import *

class leakyReluActivation(layer):
    '''
    Implementation of Leaky Rectified Linear Unit (ReLU) activation function
    '''
    def __init__(self):
        self.cache = []

    def forward(self,image):
        
        '''
        Forward propergation of leaky ReLU
        '''
        
        self.cache = image
        grad = 0.01
        return np.where(image > 0, image, image * grad)


    def backward(self,backImage):

        '''
        Backwards propergation for ReLU activation function
        '''
        
        ReLUder = np.array(backImage, copy = True)
        ReLUder[self.cache <= 0] = 0; #Deriverate of ReLU activation

        return ReLUder;

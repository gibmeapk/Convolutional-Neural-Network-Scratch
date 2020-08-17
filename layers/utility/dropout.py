from layers.layer import *

import numpy as np


class Dropout(layer):
    def __init__(self,prob,phase):
        
        '''
        Probability = Chance of dropout occuring
        self.cache = backup of the image for backpropergation
        self.phase = dropout layer should only be used during the training phase
        '''
        
        self.prob = prob
        self.cache = []
        self.phase = phase

    def forward(self, image):
        
        if(self.phase == False):
            (channel,indim) = image.shape
            drop = (np.random.rand(channel,indim) < self.prob)
            self.cache = drop
            output = (image*drop)/self.prob
            return output
        else:
            return image


    def backward(self, x):
        if(self.phase == False):
            x = (x * self.cache)/self.prob
            return x
        else:
            return x




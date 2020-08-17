import numpy as np
from layers.layer import *

class flatten(layer):
    
    '''
    Flattens the image from a 3d to 1d
    '''
    
    def __init__(self):
        self.cache = []


    def forward(self,image):
        
        '''
        Forward propergation of flattening image
        '''
        self.cache = image
        (channel,X,Y) = self.cache.shape
        flattened = image.reshape(1,channel*X*Y)
        return flattened

    def backward(self,image):
        
        '''
        Backwards propergation of 'unflattening' image
        '''
        (channel,X,Y) = self.cache.shape
        unflattened = image.reshape(channel,X,Y)
        return unflattened
        

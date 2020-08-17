import numpy as np


'''
Implementation of He normal initializer
'''

def He_initialization(num_in,num_out):
        '''
        Initalizes weights in fully-connected layer
        '''
        scale = np.sqrt(2. / num_in)
        shape = (num_in,num_out)
        return np.random.normal(0, scale, size=shape)    

def He_initalization_conv(numFilt,channels,filtSize):
        '''
        Initalizes weights in convolutional layer
        '''
        scale = np.sqrt(2./ channels)
        shape = (numFilt,channels,filtSize,filtSize)
        return np.random.normal(0,scale,size=shape)
        

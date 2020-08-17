import numpy as np

from layers.layer import *
from layers.utility.initializer import He_initalization_conv

###from matplotlib import pyplot as plt

class convolution(layer):
    
    '''
    Class is a subclass of layer, performs convolution operation
    '''
    
    def __init__(self,padding,filterSize, stride, numFilters, learningRate):
        
        '''
        initalize hyperparameters
        '''
        
        self.LEARNING_RATE = learningRate
        self.FILTER_SIZE = filterSize
        self.NUM_FILTERS = numFilters
        
        self.weights = []
        self.bias = []
        self.stride = stride
        self.padding = padding

        self.cache = []


        
    def initalizeWeight(self, channel):
        
        '''
        calls He normal initalizer (found in utility.py) to define inital weight
        '''
            
        self.weights =  He_initalization_conv(self.NUM_FILTERS,channel,self.FILTER_SIZE)


    def initalizeBias(self):
        
        '''
        sets inital bias
        '''
        
        self.bias =  np.zeros((self.NUM_FILTERS,1))



    def forward(self,image):
        
        '''
        Performs forward propergation of the convolutional layer
        '''
        
        self.cache = image
        (channel,X,Y) = self.cache.shape
           
        if len(self.weights) == 0:
            self.initalizeWeight(channel)

        if len(self.bias) == 0:
            self.initalizeBias()
        
        assert X == Y, "X and Y axis must be identical size"
        
        if self.padding != 0: #If padding
            outShape = self.pad()

        else:
            outShape = int((X - self.FILTER_SIZE)/self.stride)+ 1
            
        outDim = np.zeros((self.NUM_FILTERS,outShape,outShape))
        
        for c in range(self.NUM_FILTERS):
            for x in range(outShape):
                for y in range(outShape):
                    xS = x + self.FILTER_SIZE
                    yS = y + self.FILTER_SIZE
                    outDim[c, x, y] = (np.sum(self.weights[c] * self.cache[:,x:xS,y:yS]) + self.bias[c])                    
        return outDim


    def backward(self, backImage):
        
        '''
        Performs backwards propergation of the convolutional layer
        '''
        
        outShape = backImage.shape[1]
        
        #derivatives
        weightDeriv = np.zeros(self.weights.shape)
        biasDirv = np.zeros(self.bias.shape)
        pastDeriv = np.zeros(self.cache.shape)
        
        for c in range(self.NUM_FILTERS):
            for x in range(outShape):
                xS = x + self.FILTER_SIZE
                for y in range(outShape):
                    yS = y+self.FILTER_SIZE
                    pastDeriv[:,x:xS,y:yS] = np.add((backImage[c,x,y] * self.weights[c]),pastDeriv[:,x:xS,y:yS])
                    weightDeriv[c] = np.add((self.cache[:,x:xS,y:yS] * backImage[c,x,y]),weightDeriv[c])
            biasDirv[c] = np.sum(backImage[c])
        self.updateParams(weightDeriv,biasDirv)
        return pastDeriv
        
    def updateParams(self, w, b):
        
        '''
        Updates hyperparameters
        '''
        
        self.weights = self.weights - (self.LEARNING_RATE * w)
        self.bias = self.bias - (self.LEARNING_RATE * b)


    
    def pad(self):
        
        '''
        Pads input image with 0's if hyper parameter padding is not set to None or 0, returns calculated output shape
        '''
        padded = np.pad(self.cache, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
        outShape = int((self.cache.shape[1] - self.FILTER_SIZE + 2 * self.padding) / self.stride + 1) 

        return outDim


if __name__ == '__main__':
    img = np.random.rand(1,45,45)
    #plt.imshow(img.squeeze(), cmap='gray')
    #plt.show()

    conv = convolution(0,3,1,1,0.01)
    x = conv.forward(img)
    print(x.size)
    #plt.imshow(x.squeeze(), cmap='gray')
    #plt.show()



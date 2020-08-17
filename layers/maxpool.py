#from layers.layer import *
import numpy as np

class maxPooling():

    def __init__(self,stride,poolSize):

        '''
        initalize hyperparameters
        '''
        
        self.STRIDE = stride
        self.POOLSIZE = poolSize
        
        self.cache = []


    def forward(self, image):

        '''
        Performs forward propergation of the maxpooling layer
        '''
        
        self.cache = image
        (channel, X, _) = self.cache.shape

        assert self.cache.shape[1] == self.cache.shape[2], "X and Y axis must be identical size"
        
        outShape = int(1 + (X - self.POOLSIZE) // self.STRIDE)

        outDim = np.zeros((channel, outShape, outShape))

        for c in range(channel):
            for x in range(outShape):
                for y in range(outShape):
                    xS = x*self.STRIDE
                    yS = y*self.STRIDE
                    toDSlicee = self.cache[c, xS : xS+self.POOLSIZE, yS : yS+self.POOLSIZE]
                    outDim[c, x, y] = np.amax(toDSlicee,axis=(0, 1))
        return outDim

    def backward(self, backImage):
        
        '''
        Performs backwards propergation of the maxpooling layer
        '''
        
        (channel, X, _) = self.cache.shape
        outDim = np.zeros((channel,X,X))

        for c in range(channel):
            for x in range(X):
                if(np.mod(x,self.POOLSIZE) == 0):
                    xS = x+self.POOLSIZE 
                    for y in range(X):
                        if(np.mod(y,self.POOLSIZE) == 0):
                            yS = y+self.POOLSIZE
                            toUSample = self.cache[c, x:xS, y:yS]
                            USampleMax = np.argmax(toUSample)
                            maxX =  USampleMax//self.POOLSIZE #Index of Maximum of np column
                            maxY = USampleMax%self.POOLSIZE #Index of Maximum of np row
                            outDim[c, maxX+x, maxY+y] = backImage[c, x//self.POOLSIZE, y//self.POOLSIZE]
                    
        return outDim
    
if __name__ == '__main__':
    
    img = np.random.rand(1,46,46)
    mp = maxPooling(2,2)
    x = mp.forward(img)
    out = np.random.rand(1,23,23)
    y = mp.backward(out)
    print(y.shape)

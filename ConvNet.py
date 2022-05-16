from tqdm import tqdm
import scipy.io
import scipy.misc
from keras.utils.np_utils import to_categorical   
from imblearn.over_sampling import SMOTE


#from matplotlib import pyplot as plt


import numpy as np
from layers.convolutional import convolution
from layers.maxpool import maxPooling
from layers.softmax import *
from layers.flatten import *
from layers.leakyReLU import *
from layers.flatten import *
from layers.utility.dropout import Dropout
from layers.fullyConnected import *

from layers.utility.crossEnt import *

class MainConv:

    def __init__(self):

        self.LEARNING_RATE = 1e-4
        self.layers = [] 
        
        self.train_labels = []
        self.train_images = []
        
        self.testing_set = []
        self.test_labels = []
        
        self.trained = False

        
    def Alexnet(self):
        self.layers.append(convolution(0,5,1,32,self.LEARNING_RATE))
        self.layers.append(leakyReluActivation())
        self.layers.append(maxPooling(2,2))

        self.layers.append(convolution(0,3,1,64,self.LEARNING_RATE))
        self.layers.append(leakyReluActivation())
        self.layers.append(convolution(0,3,1,64,self.LEARNING_RATE))
        self.layers.append(leakyReluActivation())

        self.layers.append(maxPooling(2,2))

        self.layers.append(convolution(0,3,1,128,self.LEARNING_RATE))
        self.layers.append(leakyReluActivation())

        self.layers.append(convolution(0,3,1,128,self.LEARNING_RATE))
        self.layers.append(leakyReluActivation())

        self.layers.append(maxPooling(2,2))

        self.layers.append(convolution(0,3,1,256,self.LEARNING_RATE))
        self.layers.append(leakyReluActivation())

        self.layers.append(convolution(0,3,1,256,self.LEARNING_RATE))
        self.layers.append(leakyReluActivation())

        self.layers.append(maxPooling(2,2))
                
        self.layers.append(flatten())
        
        self.layers.append(fullyConnected(4096,self.LEARNING_RATE))
        self.layers.append(leakyReluActivation())
        
        self.layers.append(fullyConnected(1024,self.LEARNING_RATE))
        self.layers.append(Dropout(0.5,self.trained))
        self.layers.append(leakyReluActivation())
        
        self.layers.append(fullyConnected(3,self.LEARNING_RATE))
        self.layers.append(softmax())

    def initaliseNetwork(self):
        
        self.layers.append(convolution(0,5,1,32,self.LEARNING_RATE))
        self.layers.append(leakyReluActivation())
        self.layers.append(maxPooling(2,2))
        self.layers.append(flatten())        
        self.layers.append(fullyConnected(1024,self.LEARNING_RATE))
        self.layers.append(leakyReluActivation())
        
        self.layers.append(fullyConnected(3,self.LEARNING_RATE))
        self.layers.append(softmax())

        
    def getTrained(self):
        '''
        getter method for trained
        '''
        return self.trained

    def setTrained(self, x):
        '''
        setter method for trained
        '''
        self.trained = x

    def getTrain_Images(self):
        '''
        getter method for self.train_images
        '''
        return self.train_images

    def forward(self,x):
        '''
        method performs foward propergation of input image x
        '''
            
        for l in range(len(self.layers)): 
            output = self.layers[l].forward(x)
            x = output
        return x

        
    def backward(self,back):
        '''
        method performs backpropagation on label back
        '''
        for l in reversed(range(len(self.layers))):
            backout = self.layers[l].backward(back)
            back = backout

            
    def train(self, epoch):
        '''
        Performs the training of the neural network
        '''
        self.shuffleSets()
        acc = 0
        loss = 0
        testNumber = 0
        for j in tqdm(range(epoch)):
            for i in range(2):
                
                trainData = self.train_images[i]/255.0
                trainLabel = self.train_labels[i]
                
                output = self.forward(trainData)
                loss = loss + cross_entropy(output, trainLabel)
                if np.argmax(output) == np.argmax(trainLabel):
                    acc += 1

                back = trainLabel
                self.backward(back)
                testNumber += 1
                if(np.mod((i+1),25) == 0): #Every 25 tests performance can be monitored
                    print()
                    print("=============================")
                    print("   -Train number:",testNumber,"-   ")  
                    print("Epoch:",j)
                    print("Accuracy: ", float(acc/testNumber))
                    print("Loss: ", float(loss/testNumber))
                    print("=============================")
                    print()
            print()
            print("=============================")
            print("   -Epoch results-   ")
            print("Epoch: ",j)
            print("Accuracy: ", float(acc/testNumber))
            print("Loss: ", float(loss/testNumber))
            print("=============================")
            print()
            
        loss /= testNumber
        print("Training accuracy:", acc/testNumber)
        print("Loss: ", loss/testNumber)
        self.setTrained(True)

        
    def test(self):
        '''
        Testing phase begins on testing dataset 1/5 of total dataset
        '''
        print()
        print()
        print("TESTING...")
        print()
        data = self.test_images
        label = self.test_labels
        acc = 0
        for i in range(len(self.test_images)):
            trainData = data[i]/255.0 #divide the data by 255.0 to normalize
            trainLabel = label[i]
            for l in range(len(self.layers)):
                #Performs forward propergation
                output = self.layers[l].forward(trainData)
                
            if np.argmax(output) == np.argmax(trainLabel):#If output of softmax == output of training label then
                acc += 1 
        print("=============================")
        print("Tests: ", len(self.test_images))
        print("Accuracy: ", float(acc)/float(len(self.test_images)))


    def SMOTE(self):
        '''
        uses SMOTE to balance the dataset
        '''
        smote = SMOTE()
        (a,b,c,d) = self.train_images.shape
        x = self.train_images.reshape(a,b*c*d)
        X_sm, y_sm = smote.fit_sample(x, self.train_labels) #takes size of majority class and creates new samples for each minority classes till tehy are the size of majority
        cache = X_sm.shape[0]
        x = X_sm.reshape(cache,b,c,d)

        self.train_images = x
        self.train_labels = y_sm

        
    def shuffleSets(self):
        '''
        Shuffles data and label sets
        '''
        assert len(self.train_images) == len(self.train_labels), "Data set and labels are not the same size"
        print("Training set OK...")
        self.SMOTE()
        random = np.random.permutation(len(self.train_images))

        #shuffles training set
        ti = self.train_images[random]
        #shuffles label set
        tl = self.train_labels[random]


        #Split data into training and test
        trainSplit = round((len(self.train_images)-(len(self.train_images)*0.2))) #2/3rds training
        testSplit = round((len(self.train_images)*0.2)) #1/3rd testing
        self.train_images = ti[:trainSplit]
        self.test_images = ti[-testSplit:]

        #Split labels into training and test
        
        trainSplit = round((len(self.train_labels)-(len(self.train_labels)*0.2)))
        testSplit = round((len(self.train_labels)*0.2))

        self.train_labels = tl[:trainSplit]
        self.test_labels =  tl[-testSplit:]

        
    def trainingSet(self):
        mat = scipy.io.loadmat('Input_Z_80.mat')
        x = mat.get('Data')
        t = np.array(x).astype("float64")
        c = []
        for i in range(t.shape[3]):
            x = t[:, :, :, i]
            x = x.transpose(2,0,1)
            c.append(x)
        c = np.array(c)
        self.train_images = np.array(c)
        return self.train_images


    def importCustomMatLabel(self, file):
        '''
        takes a mat file and converts it into one hot encoding
        '''
        mat2 = scipy.io.loadmat(file[0])
        y = mat2.get('Target')
        b1 = np.array(y)
        for i in range(len(b1)):
            if b1[i] == 1.0:
                b1[i] = 2
            if b1[i] == 0.5:
                b1[i] = 1

        b1 = b1.astype(int)
        b1 = to_categorical(b1, num_classes=3) #This converts the labels to one-hot encoding from 0,1,2 to 100,010,001 
        self.train_labels =  b1
        return self.train_labels
    

    def importCustomMat(self, file):
        '''
        Takes a mat file and formats it into (size,channel,x,y)
        '''
        mat = scipy.io.loadmat(file[0])
        x = mat.get('Data')
        t = np.array(x).astype("float64")
        c = []
        for i in range(t.shape[3]):
            x = t[:, :, :, i]
            x = x.transpose(2,0,1)
            c.append(x)
        self.train_images = np.array(c)
        return self.train_images

        
    def trainingLabels(self):
        mat2 = scipy.io.loadmat('Target.mat')
        y = mat2.get('Target')
        b1 = np.array(y)
        for i in range(len(b1)):
            if b1[i] == 1.0:
                b1[i] = 2
            if b1[i] == 0.5:
                b1[i] = 1

        b1 = b1.astype(int)
        b1 = to_categorical(b1, num_classes=3)

        self.train_labels =  b1
        
        return self.train_labels

        
    def predict(self,file):
        '''
        Performs prediction on imported image
        '''
        np_image = Image.open(file[0])
        np_image = np.array(np_image)
        np_image = np_image.transpose(0,2,1)
        x = transform.resize(np_image, (1,176,176))
        output = self.forward(x)
        return output


if __name__ == '__main__':
    
    
    conv = MainConv()
    conv.initaliseNetwork()
    conv.trainingSet()
    conv.trainingLabels()
    conv.train(5)
    conv.test()




import unittest
import numpy as np
import scipy.io
import scipy.misc
from keras.utils.np_utils import to_categorical   

from layers.convolutional import convolution
from layers.maxpool import maxPooling
from layers.softmax import *
from layers.flatten import *
from layers.fullyConnected import *

class TestConvMethods(unittest.TestCase):

    def test_convForward(self):
        conv = convolution(0,3,1,1,0.01)
        img = np.random.rand(1,176,176)
        forward = conv.forward(img)
        self.assertEqual(forward.shape,(1, 174, 174))

    def test_maxPoolForward(self):
        img = np.random.rand(1,46,46)
        mp = maxPooling(2,2)
        x = mp.forward(img)
        self.assertEqual(x.shape,(1,23,23))

    def test_softmax(self):
        sm = softmax()
        img = np.random.rand(1,300)
        x = sm.forward(img)
        self.assertEqual(x.shape,(1,300))

    def test_flatten(self):
        f = flatten()
        img = np.random.rand(1,46,46)
        flat = f.forward(img)
        self.assertEqual(flat.shape,(1,(46*46)))

    def test_FC(self):
        fc = fullyConnected(256,0.01)
        img = np.random.rand(1,500)
        x = fc.forward(img)
        self.assertEqual(x.shape,(1,256))

    def test_maxPoolBackwards(self):
        img = np.random.rand(2,46,46)
        mp = maxPooling(2,2)
        x = mp.forward(img)
        out = np.random.rand(2,23,23)
        y = mp.backward(out)
        self.assertEqual(y.shape,(2,46,46))

    def test_convBackwards(self):
        conv = convolution(0,3,1,1,0.01)
        img = np.random.rand(1,176,176)
        forward = conv.forward(img)
        img = np.random.rand(1, 174, 174)
        back = conv.backward(img)
        self.assertEqual(back.shape,(1,176,176))

    def test_softMaxBackward(self):
        sm = softmax()
        img = np.random.rand(1,3)
        x = sm.forward(img)
        label = self.trainingLabels()
        image = sm.backward(label[0])
        self.assertEqual(x.shape,(1,3))

    def test_FCBackward(self):
        fc = fullyConnected(256,0.01)
        img = np.random.rand(1,500)
        x = fc.forward(img)
        img = np.random.rand(1,256)
        back = fc.backward(img)
        self.assertEqual(back.shape,(1,500))
    

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

        train_labels =  b1
        
        return train_labels
    
    def test_flatten_backward(self):
        f = flatten()
        img = np.random.rand(1,46,46)
        flat = f.forward(img)
        img = np.random.rand(1,2116)
        back = f.backward(img)
        self.assertEqual(back.shape,(1,46,46))

    def test_labelSetExtract(self):
        x = self.trainingLabels()
        assert len(x) == 178
        
        
        
if __name__ == '__main__':
    unittest.main()

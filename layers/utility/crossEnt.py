import numpy as np

def cross_entropy(image, label):
    '''
    Loss function for project
    '''
    try:
        loss = -np.sum(np.log(image)* label)
    except:
        '''
        in the event np.log(0) we allow an offset of 1e-9
        '''
        print("Log(0)")
        loss = -np.sum(np.log(image+1e-9))/image.shape[0]
    return loss

import numpy as np 
import scipy.io as sio 

f = sio.loadmat('layer1')
l1 = f['0']
l2 = f['1'][0]
l1 = np.float32(l1)
l2 = np.float32(l2)
print(l1.shape)
print(l2.shape)
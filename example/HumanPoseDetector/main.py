import network 
import numpy as np 
import tensorflow as tf 
import model3 as M 

tf.keras.backend.set_learning_phase(False)
net = network.PosePredNet(17)
saver = M.Saver(net)
saver.restore('./posedetnet/')
x = np.ones([1,256, 192, 3]).astype(np.float32)
y = net(x)
print(y[0,10,10])

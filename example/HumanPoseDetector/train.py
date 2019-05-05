import network 
import numpy as np 
import tensorflow as tf 
import model3 as M 
import data_reader

def grad_loss(x, model):
	data, label = x
	with tf.GraidentTape() as tape:
		out = model(data)
		loss = tf.reduce_mean(tf.square(out - label))
	grads = tape.gradient(loss, model.trainable_variables)
	return grads, [loss]


tf.keras.backend.set_learning_phase(False)
net = network.PosePredNet(19)
M.Saver(net.backbone).restore('./posedetnet/')
M.Saver(net.head).restore('./posedetnet/')

optim = tf.optimizers.Adam(0.0001)
saver = M.Saver(net, optim)
saver.restore('./model/')

# initialize
_ = np.zeros([1,256,256,3]).astype(np.float32)
net(_) 

# start training
reader = data_reader.data_reader()
MAX_ITER = 100000

for i in range(MAX_ITER+1):
	batch = reader.get_next()
	grads, lss = grad_loss(batch, net)
	optim.apply_gradients(M.zip_grads(grads, net.trainable_variables))
	if i%10==0:
		print('Iter:%d\tLoss:%.4f'%(i, lss[0]))
	if i%5000==0 and i>0:
		saver.save('./model/model.ckpt')

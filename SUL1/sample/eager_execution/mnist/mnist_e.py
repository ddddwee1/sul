import tensorflow as tf 
import model as M 
import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data

tf.enable_eager_execution()

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
BSIZE = 128 *2

def build_model(img_input):
	# can change whatever activation function
	with tf.variable_scope('MOD'):
		mod = M.Model(img_input,[None,28*28])
		mod.reshape([-1,28,28,1])
		mod.convLayer(5,16,activation=M.PARAM_LRELU)
		mod.maxpoolLayer(2)
		mod.convLayer(5,16,activation=M.PARAM_LRELU)
		mod.maxpoolLayer(2)
		mod.convLayer(5,16,activation=M.PARAM_LRELU)
		mod.maxpoolLayer(2)
		mod.flatten()
		mod.fcLayer(50,activation=M.PARAM_TANH)
		mod.fcLayer(10)
	return mod.get_current_layer()

def build_graph(x,y):
	with tf.GradientTape() as tape:
		last_layer = build_model(x)
		loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=last_layer))
		accuracy = M.accuracy(last_layer,tf.argmax(y,1))
	return loss,accuracy,tape

with tf.device("/gpu:0"):
	optim = tf.train.AdamOptimizer(0.001)
	for i in range(50000*2):
		x_train, y_train = mnist.train.next_batch(BSIZE)
		y_train = np.float32(y_train)
		x_train = np.float32(x_train)
		ls,acc,tape = build_graph(x_train, y_train)
		grad = tape.gradient(ls,M.VAR_LIST)
		# print(len(M.VAR_LIST))

		optim.apply_gradients(zip(grad,M.VAR_LIST))
		if i%100==0:
			print('iter:%d\tacc:%.4f\tloss:%.4f'%(i,acc,ls))

import tensorflow as tf 
import model2 as M 
import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
BSIZE = 128

class mnistNet(M.Network):
	def __init__(self):
		self.model_path = './model/'
		super().__init__()

	def build_structure(self):
		self.input_holder = tf.placeholder(tf.float32,[None, 28*28])
		self.label_holder = tf.placeholder(tf.float32,[None, 10])
		with tf.variable_scope('mnistNet'):
			mod = M.Model(self.input_holder)
			mod.reshape([-1,28,28,1])
			mod.convLayer(5,16,activation=M.PARAM_LRELU)
			mod.maxpoolLayer(2)
			mod.convLayer(5,16,activation=M.PARAM_LRELU)
			mod.maxpoolLayer(2)
			mod.convLayer(5,16,activation=M.PARAM_LRELU)
			mod.maxpoolLayer(2)
			mod.flatten()
			mod.fcLayer(50,activation=M.PARAM_TANH)
			mod.fcLayer(50,activation=M.PARAM_TANH)
			mod.fcLayer(10)
		self.logit = mod.get_current_layer()

	def build_loss(self):
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logit, labels=self.label_holder))
		self.train_op = tf.train.AdamOptimizer(0.0001).minimize(self.loss)
		self.accuracy = M.accuracy(self.logit, tf.argmax(self.label_holder, -1))

	def train(self,x ,y, norm=False):
		if norm:
			x = x * 2 - 1
		ls, acc, _ = self.sess.run([self.loss, self.accuracy, self.train_op], feed_dict= {self.input_holder: x, self.label_holder:y})
		return ls, acc

	def eval(self,x, y, norm=False):
		if norm:
			x = x * 2 - 1
		acc = self.sess.run(self.accuracy, feed_dict= {self.input_holder: x, self.label_holder:y})
		return acc

net = mnistNet()
for i in range(10000):
	x_train, y_train = mnist.train.next_batch(BSIZE)
	ls, acc = net.train(x_train, y_train)
	if i%100==0:
		print('iter',i,'\t|acc:',acc,'\tloss:',ls)
	if i%1000==0:
		acc = net.eval(mnist.test.images, mnist.test.labels)
		print('Test accuracy:',acc)
net.save('abc.ckpt')

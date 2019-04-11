import tensorflow as tf 
import modeleag as M 
import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
BSIZE = 128

class model(M.Model):
	def initialize(self):
		self.conv1 = M.ConvLayer(size=5,outchn=16,activation=M.PARAM_RELU)
		self.conv2 = M.ConvLayer(size=5,outchn=32,activation=M.PARAM_RELU)
		self.conv3 = M.ConvLayer(size=5,outchn=64,activation=M.PARAM_RELU)
		self.pool = M.maxPool(2)
		self.fc1 = M.Dense(10)

	def forward(self, x):
		x = self.conv1(x)
		x = self.pool(x)
		x = self.conv2(x)
		x = self.pool(x)
		x = self.conv3(x)
		x = self.pool(x)
		x = M.flatten(x)
		x = self.fc1(x)
		return x 

mod = model()
def loss_function(x, labels):
	with tf.GradientTape() as tape:
		logits = mod(x)
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
		accuracy = M.accuracy(logits, labels)
	return loss, accuracy, tape

optim = tf.train.AdamOptimizer(0.001)

saver = M.Saver(mod, optim)
# saver.restore('./model/')

for i in range(100):
	x_train, y_train = mnist.train.next_batch(BSIZE)
	x_train = x_train.reshape([-1, 28, 28, 1])
	loss, accuracy, tape = loss_function(x_train, y_train)
	grad = tape.gradient(loss, mod.variables)

	optim.apply_gradients(zip(grad, mod.variables))

	if i%100==0:
		print('Loss',loss.numpy(),'Acc',accuracy.numpy())

saver.save('./model/tst.ckpt')

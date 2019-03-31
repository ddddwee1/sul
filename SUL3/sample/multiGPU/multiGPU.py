import tensorflow as tf 
import layers3 as L 
import numpy as np 
import model3 as M 
import os 
import random
import glob 
import time 
import cv2 

# tf.debugging.set_log_device_placement(True)
class cnnModel(M.Model):
	def initialize(self):
		self.c1 = M.ConvLayer2D(5, 32, stride=4, activation=M.PARAM_RELU)
		self.c2 = M.ConvLayer2D(5, 32, stride=4, activation=M.PARAM_RELU)
		self.c3 = M.ConvLayer2D(5, 64, stride=4, activation=M.PARAM_RELU)
		self.fc = M.Dense(10)

	def forward(self, x):
		x = self.c1(x)
		x = self.c2(x)
		x = self.c3(x)
		x = M.flatten(x)
		x = self.fc(x)
		return x 

def get_grads(x , model):
	x, label = x 
	with tf.GradientTape() as tape:
		out = model(x)
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=label))

	grads = tape.gradient(loss, model.trainable_variables)
	return grads, loss

# try to define a generator
class data_reader():
	def __init__(self, bsize):
		self.bsize= bsize
		self.data = []
		for i in glob.glob('./data/*.*'):
			img = cv2.imread(i)
			img = np.float32(img)
			lb = np.random.randint(10)
			lb = np.eye(10)[lb].astype(np.float32)
			self.data.append([img, lb])

	def get_next(self):
		data = random.sample(self.data, self.bsize)
		dt, lb = list(zip(*data))
		dt = np.array(dt)
		lb = np.array(lb)
		return dt, lb 

reader = data_reader(512)
with tf.device('/cpu:0'):
	model = cnnModel()

	optimizer = tf.keras.optimizers.Adam(0.001)

	pt = M.ParallelTraining(model, optimizer, [0,1,2,3]) 

	for i in range(100):
		t1 = time.time()
		batch = reader.get_next()
		batch_distribute = pt.split_data(batch)
		grads, loss = pt.compute_grad_loss(batch_distribute, get_grads)
		pt.apply_grad(grads)
		t2 = time.time()
		print('LS', loss.numpy())
		print('TIME', t2-t1)

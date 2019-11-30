import numpy as np 
import tensorflow as tf 

######## Parallel Training #########
class ParallelTraining():
	# very naive implementation. Not suitable for complex structure. Will modify in the future
	def __init__(self, model, optimizer, devices, grad_loss_fn, input_size=None):
		self.model = model 
		if not input_size is None:
			_ = model(np.float32([np.ones(input_size)]))
		assert model.graph_initialized, 'Model should be initialized before parallel training'
		self.optimizer = optimizer
		self.devices = devices
		self.grad_loss_fn = grad_loss_fn

	@tf.function
	def train_step(self, data):
		with tf.device('/cpu:0'):
			rr = []
			for idx,i in enumerate(self.devices):
				with tf.device('/gpu:%d'%i):
					rr.append(self.grad_loss_fn(data[idx], self.model))
					print('Initalize GPU:%d'%i)
			losses = []
			grads = [i[0] for i in rr]
			grads = [sum(g)/len(g) for g in zip(*grads)]
			for i in rr:
				losses.append(i[1])
			self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
		return losses

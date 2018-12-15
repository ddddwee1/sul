import layers2 as L 
import tensorflow as tf 
import numpy as np 
import os 
import random

PARAM_RELU = 0
PARAM_LRELU = 1
PARAM_ELU = 2
PARAM_TANH = 3
PARAM_MFM = 4
PARAM_MFM_FC = 5
PARAM_SIGMOID = 6

######## util functions ###########
def accuracy(pred,y,name='acc'):
	with tf.variable_scope(name):
		correct = tf.equal(tf.cast(tf.argmax(pred,-1),tf.int64),tf.cast(y,tf.int64))
		acc = tf.reduce_mean(tf.cast(correct,tf.float32))
	return acc

##########################
# ETA class. I want to see the ETA. It's too boring to wait here.
class ETA():
	def __init__(self,max_value):
		self.start_time = time.time()
		self.max_value = max_value
		self.current = 0

	def start(self):
		self.start_time = time.time()
		self.current = 0

	def sec2hms(self,sec):
		hm = sec//60
		s = sec%60
		h = hm//60
		m = hm%60
		return h,m,s

	def get_ETA(self,current,is_string=True):
		self.current = current
		time_div = time.time() - self.start_time
		time_remain = time_div * float(self.max_value - self.current) / float(self.current + 1)
		h,m,s = self.sec2hms(int(time_remain))
		if is_string:
			return '%d:%d:%d'%(h,m,s)
		else:
			return h,m,s

########### model class ##########
class Model(tf.contrib.checkpoint.Checkpointable):
	def __init__(self):
		self.initialized = False
		self.initialize()

	def _gather_variables(self):
		self.variables = []
		atrs = dir(self)
		for i in atrs:
			if i[0] == '_':
				continue
			obj = getattr(self, i)
			if isinstance(obj, Model) or isinstance(obj, L.Layer):
				self.variables += obj.variables

	def get_variables(self, layers=None):
		if layers is None:
			return self.variables
		else:
			res = []
			for l in layers:
				res += l.variables
			return res 

	def __call__(self, x):
		res = self.forward(x)
		if not self.initialized:
			self._gather_variables()
		return res 

########### saver ##########
class Saver():
	def __init__(self, model, optim=None):
		self.mod = model

		self.obj = tf.contrib.checkpoint.Checkpointable()
		self.obj.m = self.mod
		self.optim = optim 
		if optim is None:
			self.ckpt = tf.train.Checkpoint(model=self.obj, optimizer_step=tf.train.get_or_create_global_step())
		else:
			self.ckpt = tf.train.Checkpoint(optimizer=optim, model=self.obj, optimizer_step=tf.train.get_or_create_global_step())
	
	def save(self, path):
		self.ckpt.save(path)

	def restore(self, path, ptype='folder'):
		if ptype=='folder':
			self.ckpt.restore(tf.train.latest_checkpoint(path))
		else:
			self.ckpt.restore(path)

######### Data Reader Template ##########
class data_reader():
	def __init__(self, one_hot=None):
		self.data_pos = 0
		self.val_pos = 0
		self.data = []
		self.val = []
		self.one_hot = False
		if one_hot is not None:
			self.one_hot = True
			self.eye = np.eye(one_hot)
		self.load_data()
		
	def get_next_batch(self,BSIZE):
		if self.data_pos + BSIZE > len(self.data):
			random.shuffle(self.data)
			self.data_pos = 0
		batch = self.data[self.data_pos : self.data_pos+BSIZE]
		x = [i[0] for i in batch]
		y = [i[1] for i in batch]
		if self.one_hot:
			y = self.eye[np.array(y)]
		self.data_pos += BSIZE
		return x,y

	def get_val_next_batch(self, BSIZE):
		if self.val_pos + BSIZE >= len(self.val):
			batch = self.val[self.val_pos:]
			random.shuffle(self.val)
			self.val_pos = 0
			is_end = True
		else:
			batch = self.data[self.data_pos : self.data_pos+BSIZE]
			is_end = False
		x = [i[0] for i in batch]
		y = [i[1] for i in batch]
		if self.one_hot:
			y = self.eye[np.array(y)]
		self.val_pos += BSIZE
		return x,y, is_end

	def get_train_iter(self, BSIZE):
		return len(self.data)//BSIZE

	def get_val_iter(self, BSIZE):
		return len(self.val)//BSIZE + 1

class list_reader():
	def __init__(self, one_hot=None):
		self.data_pos = 0
		self.val_pos = 0
		self.data = []
		self.val = []
		self.one_hot = False
		if one_hot is not None:
			self.one_hot = True
			self.eye = np.eye(one_hot)
		self.load_data()
		
	def get_next_batch(self,BSIZE):
		if self.data_pos + BSIZE > len(self.data):
			random.shuffle(self.data)
			self.data_pos = 0
		batch = self.data[self.data_pos : self.data_pos+BSIZE]
		x = [i[0] for i in batch]
		y = [i[1] for i in batch]
		if self.one_hot:
			y = self.eye[np.array(y)]
		self.data_pos += BSIZE

		x = [self.process_img(i) for i in x]
		return x,y

	def get_val_next_batch(self, BSIZE):
		if self.val_pos + BSIZE >= len(self.val):
			batch = self.val[self.val_pos:]
			random.shuffle(self.val)
			self.val_pos = 0
			is_end = True
		else:
			batch = self.data[self.data_pos : self.data_pos+BSIZE]
			is_end = False
		x = [i[0] for i in batch]
		y = [i[1] for i in batch]
		if self.one_hot:
			y = self.eye[np.array(y)]
		self.val_pos += BSIZE
		x = [self.process_img(i) for i in x]
		return x,y, is_end

	def get_train_iter(self, BSIZE):
		return len(self.data)//BSIZE

	def get_val_iter(self, BSIZE):
		return len(self.val)//BSIZE + 1

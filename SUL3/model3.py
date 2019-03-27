import layers3 as L 
import tensorflow as tf 
import numpy as np 
import tensorflow.keras.Model as KModel

# activation const
PARAM_RELU = 0
PARAM_LRELU = 1
PARAM_ELU = 2
PARAM_TANH = 3
PARAM_MFM = 4
PARAM_MFM_FC = 5
PARAM_SIGMOID = 6

######## util functions ###########
def accuracy(pred,y,name='acc', one_hot=True):
	with tf.variable_scope(name):
		if one_hot:
			correct = tf.equal(tf.cast(tf.argmax(pred,-1),tf.int64),tf.cast(tf.argmax(y,-1),tf.int64))
		else:
			correct = tf.equal(tf.cast(tf.argmax(pred,-1),tf.int64),tf.cast(y,tf.int64))
		acc = tf.reduce_mean(tf.cast(correct,tf.float32))
	return acc

################
# dumb model declaration
# make alias for init and call
class Model(KModel):
	def __init__(self, *args, **kwargs):
		super(Model, self).__init__()

		self.initialize(*args, **kwargs)

	def initialize(self, *args, **kwargs):
		pass 

	def call(self, x, *args, **kwargs):
		return self.forward(x, *args, **kwargs)

	def forward(self, x, *args, **kwargs):
		pass 

################
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

################
# Layer Class 

class ConvLayer(KModel):
	def __init__(self, size, outchn, dilation_rate=1, stride=1, pad='SAME', activation=-1, batch_norm=False, usebias=True):
		super(ConvLayer, self).__init__()
		self.conv = L.conv2D(size, outchn, stride, pad, dilation_rate, usebias)
		self.batch_norm = batch_norm
		self.activation = activation
		if batch_norm:
			self.bn = L.batch_norm()
		if activation!=-1:
			self.act = L.activation(activation)

	def call(self, x):
		x = self.conv(x)
		if self.batch_norm:
			x = self.bn(x)
		if self.activation!=-1:
			x = self.act(x)
		return x 

class ConvLayer3D(KModel):
	def __init__(self, size, outchn, dilation_rate=1, stride=1, pad='SAME', activation=-1, batch_norm=False, usebias=True):
		super(ConvLayer3D, self).__init__()
		self.conv = L.conv3D(size, outchn, stride, pad, dilation_rate, usebias)
		self.batch_norm = batch_norm
		self.activation = activation
		if batch_norm:
			self.bn = L.batch_norm
		if activation!=-1:
			self.act = L.activation(activation)

	def call(self, x):
		x = self.conv(x)
		if self.batch_norm:
			x = self.bn(x)
		if self.activation!=-1:
			x = self.act(x)
		return x 

class ConvLayer1D(KModel):
	def __init__(self, size, outchn, dilation_rate=1, stride=1, pad='SAME', activation=-1, batch_norm=False, usebias=True):
		super(ConvLayer1D, self).__init__()
		self.conv = L.conv1D(size, outchn, stride, pad, dilation_rate, usebias)
		self.batch_norm = batch_norm
		self.activation = activation
		if batch_norm:
			self.bn = L.batch_norm
		if activation!=-1:
			self.act = L.activation(activation)

	def call(self, x):
		x = self.conv(x)
		if self.batch_norm:
			x = self.bn(x)
		if self.activation!=-1:
			x = self.act(x)
		return x 

class DeconvLayer1D(KModel):
	def __init__(self, size, outchn, dilation_rate=1, stride=1, pad='SAME', activation=-1, batch_norm=False, usebias=True):
		super(DeconvLayer1D, self).__init__()
		self.conv = L.deconv1D(size, outchn, stride, pad, dilation_rate, usebias)
		self.batch_norm = batch_norm
		self.activation = activation
		if batch_norm:
			self.bn = L.batch_norm
		if activation!=-1:
			self.act = L.activation(activation)

	def call(self, x):
		x = self.conv(x)
		if self.batch_norm:
			x = self.bn(x)
		if self.activation!=-1:
			x = self.act(x)
		return x 

class DeconvLayer(KModel):
	def __init__(self, size, outchn, dilation_rate=1, stride=1, pad='SAME', activation=-1, batch_norm=False, usebias=True):
		super(DeconvLayer, self).__init__()
		self.conv = L.deconv2D(size, outchn, stride, pad, dilation_rate, usebias)
		self.batch_norm = batch_norm
		self.activation = activation
		if batch_norm:
			self.bn = L.batch_norm
		if activation!=-1:
			self.act = L.activation(activation)

	def call(self, x):
		x = self.conv(x)
		if self.batch_norm:
			x = self.bn(x)
		if self.activation!=-1:
			x = self.act(x)
		return x 

class DeconvLayer3D(KModel):
	def __init__(self, size, outchn, dilation_rate=1, stride=1, pad='SAME', activation=-1, batch_norm=False, usebias=True):
		super(DeconvLayer3D, self).__init__()
		self.conv = L.deconv3D(size, outchn, stride, pad, dilation_rate, usebias)
		self.batch_norm = batch_norm
		self.activation = activation
		if batch_norm:
			self.bn = L.batch_norm
		if activation!=-1:
			self.act = L.activation(activation)

	def call(self, x):
		x = self.conv(x)
		if self.batch_norm:
			x = self.bn(x)
		if self.activation!=-1:
			x = self.act(x)
		return x 

class Dense(KModel):
	def __init__(self, outsize, batch_norm=False, activation=-1 , usebias=True):
		super(Dense, self).__init__()
		self.fc = L.fcLayer(outsize, usebias)
		self.batch_norm = batch_norm
		self.activation = activation
		if batch_norm:
			self.bn = L.batch_norm()
		if activation!=-1:
			self.act = L.activation(activation)

	def call(self, x):
		x = self.fc(x)
		if self.batch_norm:
			x = self.bn(x)
		if self.activation!=-1:
			x = self.act(x)
		return x 

class GraphConvLayer(KModel):
	def __init__(self, outsize, adj_mtx=None, adj_fn=None, usebias=True, activation=-1, batch_norm=False):
		super(GraphConvLayer, self).__init__()
		self.GCL = L.graphConvLayer(outsize, adj_mtx=adj_mtx, adj_fn=adj_fn, usebias=usebias)
		self.batch_norm = batch_norm
		self.activation = activation
		if batch_norm:
			self.bn = L.batch_norm()
		if activation!=-1:
			self.act = L.activation(activation)

	def call(self, x):
		x = self.GCL(x)
		if self.batch_norm:
			x = self.bn(x)
		if self.activation!=-1:
			x = self.act(x)
		return x 

###############
# alias for layers
AvgPool = L.avgpoolLayer
MaxPool = L.maxpoolLayer
flatten = L.flatten()

###############
# Saver 
class Saver():
	def __init__(self, model=None):
		self.model = model 

	def save(self, path):
		self.model.save_weights(path)

	def restore(self, path):
		try:
			if self.model is None:
				return tf.keras.models.load_model(path)
			else:
				if not path[-4:]=='ckpt':
					last_ckpt = tf.train.latest_checkpoint(path)
					if last_ckpt is None:
						print('No model found in checkpoint.')
						print('Model will auto-initialize after first iteration.')
						return 
					else:
						path = last_ckpt
				self.model.load_weights(path)
		except Exception as e :
			print(e)
			print('Model restore failed.')
			print('Model will auto-initialize after first iteration.')

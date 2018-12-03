import tensorflow as tf 
import numpy as np 


###########################################################
#define weight and bias initialization
def weight(shape,dtype=None):
	w = tf.get_variable('weight',shape,initializer=tf.contrib.layers.xavier_initializer(),dtype=dtype)
	return w

def weight_conv(shape,dtype=None):
	k = tf.get_variable('kernel',shape,initializer=tf.contrib.layers.xavier_initializer_conv2d(),dtype=dtype)
	return k 

def bias(shape,value=0.0,dtype=None):
	b = tf.get_variable('bias',shape,initializer=tf.constant_initializer(value),dtype=dtype)
	return b

###########################################################
#define layer class
class Layer():
	def __init__(self, name):
		# template for layer definition
		if not name is None:
			with tf.variable_scope(name):
				self._parse_args()
				self._initialize()
				self.output = self._deploy()
		else:
			self._parse_args()
			self._initialize()
			self.output = self._deploy()

	def _add_variable(self,var):
		if not self.variables:
			self.variables = []
		self.variables.append(var)

	def _initialize(self):
		pass

	def _parse_args(self):
		pass

###########################################################
#define basic layers

class conv2D(Layer):
	def __init__(self, x,size,outchn,name=None,stride=1,pad='SAME',usebias=True,kernel_data=None,bias_data=None,dilation_rate=1,weight_norm=False):
		self.x = x
		self.size = size
		self.outchn = outchn
		self.name = name
		self.stride = stride
		self.pad = pad 
		self.usebias = usebias
		self.kernel_data = kernel_data
		self.bias_data = bias_data
		self.dilation_rate = dilation_rate
		self.weight_norm = weight_norm

		super.__init__(name)

	def _parse_args(self):
		# set size
		inchannel = self.x.get_shape().as_list()[-1]
		if isinstance(self.size,list):
			self.size = [self.size[0],self.size[1],inchannel,self.outchn]
		else:
			self.size = [self.size, self.size, inchannel, self.outchn]
		# set stride
		if isinstance(self.stride,list):
			self.stride = [1,self.stride[0],self.stride[1],1]
		else:
			self.stride = [1,self.stride, self.stride, 1]
		# set dilation
		if isinstance(self.dilation_rate,list):
			self.dilation_rate = [1,self.dilation_rate[0],self.dilation_rate[1],1]
		else:
			self.dilation_rate = [1,self.dilation_rate,self.dilation_rate,1]

	def _initialize(self):
		# this will enlarge ckpt size. (at first time)
		if self.kernel_data:
			self.W = tf.constant(self.kernel_data, name='kernel')
		else:
			self.W = weight_conv(self.size)
			if self.weight_norm:
				print('Enable weight norm')
				self.W = self.W.initialized_value()
				self.W = tf.nn.l2_normalize(self.W, [0,1,2])
				print('Initialize weight norm')
				x_init = tf.nn.conv2d(self.x,self.W,stride,pad,dilations=dilation_rate)
				m_init, v_init = tf.nn.moments(x_init,[0,1,2])
				s_init = 1. / tf.sqrt(v_init + 1e-8)
				s = tf.get_variable('weight_scale',dtype=tf.float32,initializer=s_init)
				self.S = s.initialized_value()
				self.S = tf.reshape(self.S,[1,1,1,outchn])
				self.W = self.S *self.W
				self._add_variable(self.S)
		self._add_variable(self.W)

		# 
		if self.usebias:
			if self.bias_data:
				self.b = bias([self.outchn], self.bias_data)
			else:
				self.b = bias([self.outchn])
		self._add_variable(self.b)
		
	def _deploy(self):
		out = tf.nn.conv2d(x,w,stride,pad,dilations=dilation_rate)
		if self.b:
			out = tf.nn.bias_add(out,b)
		return out 

class maxpoolLayer(Layer):
	def __init__(self,x,size,stride=None,name=None,pad='SAME'):
		self.x = x 
		self.name = name
		self.stride = stride
		self.pad = pad

		super.__init__(name)

	def _parse_args(self):
		if isinstance(self.size, list):
			if len(self.size)==2:
				self.size = [1, self.size[0], self.size[1], 1]
		elif isinstance(self.size, int):
			self.size = [1, self.size, self.size, 1]

		if not self.stride:
			self.stride = self.size
		elif isinstance(self.stride, list):
			if len(self.stride)==2:
				self.stride = [1,self.stride[0],self.stride[1],1]
		elif isinstance(self.stride, int):
			self.stride = [1, self.stride, self.stride, 1]

	def _deploy(self):
		return tf.nn.max_pool(self.x, ksize=self.size, strides=self.stride, padding=self.pad)

class activation(Layer):
	def __init__(self, x, param, name=None, **kwarg):
		self.x = x 
		self.param = param
		self.name = name
		self.kwarg = kwarg

		super.__init__(name)

	def _deploy(self):
		if self.param == 0:
			res =  tf.relu(self.x)
		elif self.param == 1:
			if 'leaky' in self.kwarg:
				leaky = self.kwarg['leaky']
			else:
				leaky = 0.2
			res =  tf.maximum(self.x,self.x*leaky)
		elif self.param == 2:
			res =  tf.elu(self.x)
		elif self.param == 3:
			res =  tf.tanh(self.x)
		elif self.param == 4:
			shape = self.x.get_shape().as_list()
			res = tf.reshape(self.x,[-1,shape[1],shape[2],2,shape[-1]//2])
			res = tf.reduce_max(res,axis=[3])
		elif self.param == 5:
			shape = self.x.get_shape().as_list()
			res = tf.reduce_max(tf.reshape(self.x,[-1,2,shape[-1]//2]),axis=[1])
		elif self.param == 6:
			res =  tf.sigmoid(self.x)
		else:
			res =  self.x
		return res

class fcLayer(layer):
	def __init__(self, x, outsize, usebias, name=None):
		self.x = x 
		self.outsize = outsize
		self.usebias = usebias
		self.name = name 

		super().__init__(name)

	def _initialize(self):
		insize = self.x.get_shape().as_list()[-1]
		self.W = weight([insize, outsize])
		self._add_variable(self.W)
		if self.usebias:
			self.b = bias([outsize])
			self._add_variable(self.b)

	def _deploy(self):
		res = tf.matmul(self.x, self.W)
		if self.usebias:
			res = self.bias_add(self.res, self.b)
		return res 

class batch_norm(Layer):
	def __init__(self, x, training, epsilon, name=None):
		self.x = x 
		self.training = training
		self.epsilon = epsilon
		self.name = name

		super().__init__(name)

	def _deploy(self):
		# will modify this to lower api in later version
		if not epsilon is None:
			return tf.layers.batch_normalization(inp,training=training,name=name,epsilon=epsilon)
		return tf.layers.batch_normalization(inp,training=training,name=name)

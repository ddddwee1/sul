import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf 
from tensorflow.keras.layers import Layer as KLayer 
import numpy as np 
import time
import helper
import pickle 

params_dict = {}

def init_caffe_input(x):
	global caffe_string, layer_counter
	if not 'caffe_string' in globals():
		caffe_string = ''
	if not 'layer_counter' in globals():
		layer_counter = 0
	caffe_string += 'layer{\n'
	caffe_string += ' name: "%s"\n'%x[1]()
	caffe_string += ' type: "Input"\n'
	caffe_string += ' top: "%s"\n'%x[1]()
	caffe_string += ' input_param{\n  shape{\n   dim:%d\n   dim:%d\n   dim:%d\n   dim:%d\n  }\n }\n}\n'%(x[0].shape[0], x[0].shape[3], x[0].shape[1], x[0].shape[2])
	layer_counter += 1 

# def pad_correction(x, conv_layer):
# 	# TF padding is shifted by 1 compared to caffe
# 	# we achieve this by creating a dummy layer
# 	global caffe_string, layer_counter
# 	if not 'caffe_string' in globals():
# 		caffe_string = ''
# 	if not 'layer_counter' in globals():
# 		layer_counter = 0

# 	layer_name = 'dummy%d'%layer_counter
# 	caffe_string += 'layer{\n'
# 	caffe_string += ' name: "%s"\n'%layer_name
# 	caffe_string += ' type: "Input"\n'
# 	caffe_string += ' top: "%s"\n'%layer_name
# 	caffe_string += ' input_param{\n  shape{\n   dim:%d\n   dim:%d\n   dim:%d\n   dim:%d\n  }\n }\n}\n'%(x.shape[0], x.shape[3], x.shape[1], x.shape[2])
# 	layer_name0 = layer_name

# 	layer_name = 'crop%d'%layer_counter
# 	caffe_string += 'layer{\n'
# 	caffe_string += ' name: "%s"\n'%layer_name
# 	caffe_string += ' type: "Crop"\n'
# 	caffe_string += ' bottom: "%s"\n'%conv_layer
# 	caffe_string += ' bottom: "%s"\n'%layer_name0
# 	caffe_string += ' top: "%s"\n'%layer_name
# 	caffe_string += ' crop_param{\n  offset:%d\n  offset:%d\n  }\n}\n'%(1,1)
# 	return layer_name

def pad_correction(x, conv_layer):
	# TF padding is shifted by 1 compared to caffe 
	# We dont have dummy data and cropping layers. 
	# We achieve this by incorporating a 2x2 depthwise conv 

	def get_kernel(outchn):
		res = np.zeros([2,2, outchn, 1]).astype(np.float32)
		for i in range(outchn):
			res[1,1,i] = 1
		return res 

	global caffe_string, layer_counter
	if not 'caffe_string' in globals():
		caffe_string = ''
	if not 'layer_counter' in globals():
		layer_counter = 0

	layer_name = 'padshift%d'%layer_counter

	outchn = x.shape[-1]

	caffe_string += 'layer{\n'
	caffe_string += ' name: "%s"\n'%layer_name
	caffe_string += ' type: "Convolution"\n'
	caffe_string += ' bottom: "%s"\n'%conv_layer
	caffe_string += ' top: "%s"\n'%layer_name
	caffe_string += ' convolution_param{\n'
	caffe_string += '  num_output: %d\n'%outchn
	caffe_string += '  bias_term: %s\n'%('false')
	caffe_string += '  group: %d\n'%outchn
	caffe_string += '  stride: 1\n'
	caffe_string += '  pad_h: 0\n'
	caffe_string += '  pad_w: 0\n'
	caffe_string += '  kernel_h: 2\n'
	caffe_string += '  kernel_w: 2\n'
	caffe_string += ' }\n}\n'

	params_dict[layer_name] = {}
	params_dict[layer_name]['dwkernel'] = get_kernel(outchn)
	return layer_name

def save_params(name):
	pickle.dump(params_dict, open(name, 'wb'))

# dumb layer declaration
class Layer(KLayer):
	"""
	Layer template. Implement some basic functions by override initialize, build and forward.
	"""
	def __init__(self, *args, **kwargs):
		"""
		Default initialization. Not recommended to touch.
		"""
		super(Layer, self).__init__()
		self.initialize(*args, **kwargs)

	def initialize(self, *args, **kwargs):
		"""
		Write initialization logic here.

		This is a method to assign pre-defined parameters to the class.
		"""
		pass 

	def build(self, input_shape):
		pass 

	def call(self, x, *args, **kwargs):
		return self.forward(x, *args, **kwargs)

	def forward(self, x, *args, **kwargs):
		"""
		Alternative for *call*.

		:param x: Input tensor or numpy array. The object will be automatically converted to tensor if the input is np.array. Note that other arrays in args or kwargs will not be auto-converted.
		"""
		pass 

class conv2D(KLayer):
	"""
	Basic convolution 2D layer
	"""
	def __init__(self, size, outchn, stride=1,pad='SAME',dilation_rate=1,usebias=True,values=None):
		"""
		:type size: int or list[int]
		:param size: Indicate the size of convolution kernel.

		:type outchn: int
		:param outchn: Number of output channels

		:type stride: int or list[int]
		:param stride: Stride number. Can be either integer or list of integers

		:type pad: String
		:param pad: Padding method, must be one of 'SAME', 'VALID', 'SAME_LEFT'. 'VALID' does not use auto-padding scheme. 'SAME' uses tensorflow-style auto-padding and 'SAME_LEFT' uses pytorch-style auto-padding.

		:type dilation_rate: int or list[int]
		:param dilation_rate: Dilation rate. Can be either integer or list of integers. When dilation_rate is larger than 1, stride should be 1.

		:type usebias: bool
		:param usebias: Whether to add bias term in this layer.

		:type values: list[np.array]
		:param values: If the param 'values' is set, the layer will be initialized with the list of numpy array.
		"""
		super(conv2D, self).__init__()
		self.size = size
		self.outchn = outchn
		self.stride = stride
		self.usebias = usebias
		self.values = values
		self.dilation_rate = dilation_rate
		assert (pad in ['SAME','VALID','SAME_LEFT'])
		self.pad = pad 

	def _parse_args(self, input_shape):
		inchannel = input_shape[0][-1]
		# parse args
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

	def build(self, input_shape):
		values = self.values
		self._parse_args(input_shape)
		if self.values is not None:
			self.kernel = self.add_variable('kernel', shape=self.size, initializer=tf.initializers.constant(values[0]))
		else:
			self.kernel = self.add_variable('kernel', shape=self.size, initializer=tf.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal'))
		if self.usebias:
			if self.values is not None:
				self.bias = self.add_variable('bias', shape=[self.outchn], initializer=tf.initializers.constant(values[1]))
			else:
				self.bias = self.add_variable('bias', shape=[self.outchn], initializer=tf.initializers.constant(0.0))
		if self.pad == 'SAME_LEFT':
			self.pad_value = [self.size[0]//2, self.size[1]//2]

	def _write_caffe(self, name, out):
		global caffe_string, layer_counter
		if not 'caffe_string' in globals():
			caffe_string = ''
		if not 'layer_counter' in globals():
			layer_counter = 0
		layer_name = 'conv%d'%layer_counter

		stride = self.stride[1]
		if stride==1:
			pad = self.size[0]//2
		else:
			pad = self.size[0]//2 + 1
		caffe_string += 'layer{\n'
		caffe_string += ' name: "%s"\n'%layer_name
		caffe_string += ' type: "Convolution"\n'
		caffe_string += ' bottom: "%s"\n'%name()
		caffe_string += ' top: "%s"\n'%layer_name
		caffe_string += ' convolution_param{\n'
		caffe_string += '  num_output: %d\n'%self.outchn
		caffe_string += '  bias_term: %s\n'%('true' if self.usebias else 'false')
		caffe_string += '  group: 1\n'
		caffe_string += '  stride: %d\n'%stride
		caffe_string += '  pad_h: %d\n'%pad
		caffe_string += '  pad_w: %d\n'%pad
		caffe_string += '  kernel_h: %d\n'%(self.size[0])
		caffe_string += '  kernel_w: %d\n'%(self.size[1])
		caffe_string += ' }\n}\n'

		params_dict[layer_name] = {}
		params_dict[layer_name]['kernel'] = self.kernel.numpy()
		if self.usebias:
			params_dict[layer_name]['bias'] = self.bias.numpy()

		if stride>1:
			layer_name = pad_correction(out, layer_name)

		layer_counter += 1 
		return helper.LayerName(layer_name)

	def call(self, x):
		"""
		:param x: Input tensor or numpy array. The object will be automatically converted to tensor if the input is np.array. Note that other arrays in args or kwargs will not be auto-converted.
		"""
		name = x[1]
		x = x[0]
		if self.pad=='SAME_LEFT':
			x = tf.pad(x, [[0,0], [self.pad_value[0], self.pad_value[0]], [self.pad_value[1], self.pad_value[1]], [0,0]])
			pad = 'VALID'
		else:
			pad = self.pad
		out = tf.nn.conv2d(x, self.kernel, self.stride, pad, dilations=self.dilation_rate)
		if self.usebias:
			out = tf.nn.bias_add(out, self.bias)
		lname = self._write_caffe(name, out)
		return out, lname

class dwconv2D(KLayer):
	"""
	Basic depth-wise convolution layer.
	"""
	def __init__(self, size, multiplier, stride=1,pad='SAME',dilation_rate=1,usebias=True,values=None):
		"""
		:type size: int or list[int]
		:param size: Indicate the size of convolution kernel.

		:type multiplier: int
		:param multiplier: Multiplier of number of output channel. (outchannel = multiplier * inchannel)

		:type stride: int or list[int]
		:param stride: Stride number. Can be either integer or list of integers

		:type pad: String
		:param pad: Padding method, must be one of 'SAME', 'VALID', 'SAME_LEFT'. 'VALID' does not use auto-padding scheme. 'SAME' uses tensorflow-style auto-padding and 'SAME_LEFT' uses pytorch-style auto-padding.

		:type dilation_rate: int or list[int]
		:param dilation_rate: Dilation rate. Can be either integer or list of integers. When dilation_rate is larger than 1, stride should be 1.

		:type usebias: bool
		:param usebias: Whether to add bias term in this layer.

		:type values: list[np.array]
		:param values: If the param 'values' is set, the layer will be initialized with the list of numpy array.
		"""
		super(dwconv2D, self).__init__()
		self.size = size
		self.multiplier = multiplier
		self.stride = stride
		self.usebias = usebias
		self.values = values
		self.dilation_rate = dilation_rate
		assert (pad in ['SAME','VALID','SAME_LEFT'])
		self.pad = pad 

	def _parse_args(self, input_shape):
		inchannel = input_shape[0][-1]
		self.inchannel = inchannel
		self.outchn = inchannel * self.multiplier
		# parse args
		if isinstance(self.size,list):
			self.size = [self.size[0],self.size[1],inchannel,self.multiplier]
		else:
			self.size = [self.size, self.size, inchannel, self.multiplier]
		# set stride
		if isinstance(self.stride,list):
			self.stride = [1,self.stride[0],self.stride[1],1]
		else:
			self.stride = [1,self.stride, self.stride, 1]
		# set dilation
		if isinstance(self.dilation_rate,list):
			self.dilation_rate = [self.dilation_rate[0],self.dilation_rate[1]]
		else:
			self.dilation_rate = [self.dilation_rate,self.dilation_rate]

	def build(self, input_shape):
		values = self.values
		self._parse_args(input_shape)
		if self.values is not None:
			self.kernel = self.add_variable('kernel', shape=self.size, initializer=tf.initializers.constant(values[0]))
		else:
			self.kernel = self.add_variable('kernel', shape=self.size, initializer=tf.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal'))
		if self.usebias:
			if self.values is not None:
				self.bias = self.add_variable('bias', shape=[self.outchn], initializer=tf.initializers.constant(values[1]))
			else:
				self.bias = self.add_variable('bias', shape=[self.outchn], initializer=tf.initializers.constant(0.0))
		if self.pad == 'SAME_LEFT':
			self.pad_value = [self.size[0]//2, self.size[1]//2]

	def _write_caffe(self, name, out):
		global caffe_string, layer_counter
		if not 'caffe_string' in globals():
			caffe_string = ''
		if not 'layer_counter' in globals():
			layer_counter = 0
		layer_name = 'conv%d'%layer_counter

		stride = self.stride[1]
		if stride==1:
			pad = self.size[0]//2
		else:
			pad = self.size[0]//2 + 1
		caffe_string += 'layer{\n'
		caffe_string += ' name: "%s"\n'%layer_name
		caffe_string += ' type: "Convolution"\n'
		caffe_string += ' bottom: "%s"\n'%name()
		caffe_string += ' top: "%s"\n'%layer_name
		caffe_string += ' convolution_param{\n'
		caffe_string += '  num_output: %d\n'%self.outchn
		caffe_string += '  bias_term: %s\n'%('true' if self.usebias else 'false')
		caffe_string += '  group: %d\n'%self.inchannel
		caffe_string += '  stride: %d\n'%stride
		caffe_string += '  pad_h: %d\n'%pad
		caffe_string += '  pad_w: %d\n'%pad
		caffe_string += '  kernel_h: %d\n'%(self.size[0])
		caffe_string += '  kernel_w: %d\n'%(self.size[1])
		caffe_string += ' }\n}\n'

		params_dict[layer_name] = {}
		params_dict[layer_name]['dwkernel'] = self.kernel.numpy()
		if self.usebias:
			params_dict[layer_name]['bias'] = self.bias.numpy()

		if stride>1:
			layer_name = pad_correction(out, layer_name)

		layer_counter += 1 
		return helper.LayerName(layer_name)

	def call(self, x):
		"""
		:param x: Input tensor or numpy array. The object will be automatically converted to tensor if the input is np.array. Note that other arrays in args or kwargs will not be auto-converted.
		"""
		name = x[1]
		x = x[0]
		if self.pad=='SAME_LEFT':
			x = tf.pad(x, [[0,0], [self.pad_value[0], self.pad_value[0]], [self.pad_value[1], self.pad_value[1]], [0,0]])
			pad = 'VALID'
		else:
			pad = self.pad

		out = tf.nn.depthwise_conv2d(x, self.kernel, self.stride, pad, dilations=self.dilation_rate)
		if self.usebias:
			out = tf.nn.bias_add(out, self.bias)
		lname = self._write_caffe(name, out)
		return out, lname

class globalAvgpoolLayer(KLayer):
	"""
	Basic global average pooling layer
	"""
	def __init__(self):
		super(globalAvgpoolLayer, self).__init__()

	def build(self, input_shape):
		self.num_dim = len(input_shape[0])
		self.ksize = input_shape[0][1]

	def _write_caffe(self, name):
		global caffe_string, layer_counter
		if not 'caffe_string' in globals():
			caffe_string = ''
		if not 'layer_counter' in globals():
			layer_counter = 0
		layer_name = 'gavgpool%d'%layer_counter

		caffe_string += 'layer{\n'
		caffe_string += ' name: "%s"\n'%layer_name
		caffe_string += ' type: "Pooling"\n'
		caffe_string += ' bottom:"%s"\n'%name()
		caffe_string += ' top: "%s"\n'%layer_name
		caffe_string += ' pooling_param{\n  pool:AVE\n  kernel_size:%d\n }\n'%self.ksize
		caffe_string += '}\n'
		return helper.LayerName(layer_name)

	def call(self, x):
		"""
		:param x: Input tensor or numpy array. The object will be automatically converted to tensor if the input is np.array. Note that other arrays in args or kwargs will not be auto-converted.
		"""
		name = x[1]
		x = x[0]
		if self.num_dim==3:
			res = tf.reduce_mean(x, axis=1, keepdims=True)
		elif self.num_dim==4:
			res = tf.reduce_mean(x, axis=[1,2], keepdims=True)
		elif self.num_dim==5:
			res = tf.reduce_mean(x, axis=[1,2,3], keepdims=True)
		lname = self._write_caffe(name)
		return res , lname

class activation(KLayer):
	"""
	Basic activation layer
	"""
	def __init__(self, param, **kwargs):
		"""
		Possible values:
			- model3.PARAM_RELU
			- model3.PARAM_LRELU
			- model3.PARAM_ELU
			- model3.PARAM_TANH
			- model3.PARAM_MFM
			- model3.PARAM_MFM_FC
			- model3.PARAM_SIGMOID
			- model3.PARAM_SWISH
		"""
		super(activation, self).__init__()

		self.param = param
		self.kwargs = kwargs

	def _write_caffe(self, btm):
		global caffe_string, layer_counter
		if not 'caffe_string' in globals():
			caffe_string = ''
		if not 'layer_counter' in globals():
			layer_counter = 0
		layer_name = 'actv%d_%d'%(layer_counter, self.param)

		caffe_string += 'layer{\n'
		caffe_string += ' name: "%s"\n'%layer_name
		if self.param == 0:
			caffe_string += ' type: "ReLU"\n'
		elif self.param == 1:
			caffe_string += ' type: "PReLU"\n'
			params_dict[layer_name] = {}
			params_dict[layer_name]['gamma'] = 0.2
		elif self.param == 6:
			caffe_string += ' type: "Sigmoid"\n'
		caffe_string += ' bottom: "%s"\n'%btm()
		caffe_string += ' top: "%s"\n'%btm()
		caffe_string += '}\n'


		layer_counter += 1 
		return btm 

	def call(self, x):
		"""
		:param x: Input tensor or numpy array. The object will be automatically converted to tensor if the input is np.array. Note that other arrays in args or kwargs will not be auto-converted.
		"""
		name = x[1]
		x = x[0]
		if self.param == 0:
			res =  tf.nn.relu(x)
		elif self.param == 1:
			if 'leaky' in self.kwargs:
				leaky = self.kwargs['leaky']
			else:
				leaky = 0.2
			res =  tf.maximum(x,x*leaky)
		elif self.param == 2:
			res =  tf.nn.elu(x)
		elif self.param == 3:
			res =  tf.tanh(x)
		elif self.param == 4:
			shape = x.get_shape().as_list()
			res = tf.reshape(x,[-1,shape[1],shape[2],2,shape[-1]//2]) # potential bug in conv_net
			res = tf.reduce_max(res,axis=[3])
		elif self.param == 5:
			shape = x.get_shape().as_list()
			res = tf.reduce_max(tf.reshape(x,[-1,2,shape[-1]//2]),axis=[1])
		elif self.param == 6:
			res =  tf.sigmoid(x)
		elif self.param == 7:
			# res = tf.nn.swish(x)
			# res = tf.sigmoid(x) * x 
			res = swish(x)
		else:
			res =  x
		lname = self._write_caffe(name)
		return res, lname

class fcLayer(KLayer):
	"""
	Basic fully connected layer
	"""
	def __init__(self, outsize, usebias=True, values=None, norm=False, map_shape=None):
		"""
		:type outsize: int
		:param outsize: Number of output channels

		:type usebias: bool
		:param usebias: Whether to add bias term in this layer.

		:type values: list[np.array]
		:param values: If the param 'values' is set, the layer will be initialized with the list of numpy array.

		:type norm: bool (default=False)
		:param norm: Whether to normalize the kernel (along axis 0) before matrix multiplication

		:type map_shape: list (default=None)
		:param map_shape: If shape is set, weight will be re-shaped to fit NCHW format
		"""
		super(fcLayer, self).__init__()
		self.outsize = outsize
		self.usebias = usebias
		self.values = values
		self.norm = norm 
		self.map_shape = map_shape

	def _parse_args(self, input_shape):
		# set size
		insize = input_shape[0][-1]
		self.size = [insize, self.outsize]

	def build(self, input_shape):
		values = self.values
		self._parse_args(input_shape)
		if self.values is not None:
			self.kernel = self.add_variable('kernel', shape=self.size, initializer=tf.initializers.constant(values[0]))
		else:
			self.kernel = self.add_variable('kernel', shape=self.size, initializer=tf.initializers.GlorotUniform())
		if self.usebias:
			if self.values is not None:
				self.bias = self.add_variable('bias', shape=[self.outsize], initializer=tf.initializers.constant(values[1]))
			else:
				self.bias = self.add_variable('bias', shape=[self.outsize], initializer=tf.initializers.constant(0.0))

	def _write_caffe(self, name):
		global caffe_string, layer_counter
		if not 'caffe_string' in globals():
			caffe_string = ''
		if not 'layer_counter' in globals():
			layer_counter = 0
		layer_name = 'fc%d'%layer_counter

		caffe_string += 'layer{\n'
		caffe_string += ' name: "%s"\n'%layer_name
		caffe_string += ' type: "InnerProduct"\n'
		caffe_string += ' bottom: "%s"\n'%name()
		caffe_string += ' top: "%s"\n'%layer_name
		caffe_string += ' inner_product_param{\n'
		caffe_string += '  num_output: %d\n'%self.outsize
		caffe_string += '  bias_term: %s\n'%('true' if self.usebias else 'false')
		caffe_string += ' }\n}\n'

		params_dict[layer_name] = {}
		if self.map_shape is None:
			params_dict[layer_name]['fckernel'] = self.kernel.numpy()
		else:
			transpose_w = self.kernel.numpy()
			transpose_w = np.reshape(transpose_w, [self.map_shape[0], self.map_shape[1], self.map_shape[2], self.outsize])
			transpose_w = np.transpose(transpose_w, [2,1,0,3])
			transpose_w = np.reshape(transpose_w, [-1, self.outsize])
			params_dict[layer_name]['fckernel'] = transpose_w
		if self.usebias:
			params_dict[layer_name]['bias'] = self.bias.numpy()

		layer_counter += 1 
		return helper.LayerName(layer_name)

	def call(self, x):
		"""
		:param x: Input tensor or numpy array. The object will be automatically converted to tensor if the input is np.array. Note that other arrays in args or kwargs will not be auto-converted.
		"""
		name = x[1]
		x = x[0]
		if self.norm:
			k = tf.nn.l2_normalize(self.kernel, axis=0)
		else:
			k = self.kernel
		res = tf.matmul(x, k)
		if self.usebias:
			res = tf.nn.bias_add(res, self.bias)
		lname = self._write_caffe(name)
		return res, lname

class batch_norm(KLayer):
	"""
	Basic batch normalization layer
	"""
	def __init__(self, decay=1e-2, epsilon=1e-5, is_training=None, values=None):
		"""
		:type decay: float
		:param decay: Decay rate.

		:type epsilon: float
		:param epsilon: Epsilon value to avoid 0 division.

		:type is_training: bool
		:param is_training: Define whether this layer is in training mode

		:type values: list[np.array]
		:param values: If the param 'values' is set, the layer will be initialized with the list of numpy array.
		"""
		super(batch_norm, self).__init__()

		self.decay = decay
		self.epsilon = epsilon
		self.is_training = is_training
		self.values = values

	def build(self, input_shape):
		values = self.values
		shape = input_shape[-1]
		if self.values is None:
			self.moving_average = self.add_variable('moving_average',[shape],initializer=tf.initializers.constant(0.0),trainable=False)
			self.variance = self.add_variable('variance',[shape],initializer=tf.initializers.constant(1.0),trainable=False)

			self.gamma = self.add_variable('gamma',[shape],initializer=tf.initializers.constant(1.0),trainable=True)
			self.beta = self.add_variable('beta',[shape],initializer=tf.initializers.constant(0.0),trainable=True)
		else:
			self.moving_average = self.add_variable('moving_average',[shape],initializer=tf.initializers.constant(self.values[0]),trainable=False)
			self.variance = self.add_variable('variance',[shape],initializer=tf.initializers.constant(values[1]),trainable=False)

			self.gamma = self.add_variable('gamma',[shape],initializer=tf.initializers.constant(values[2]),trainable=True)
			self.beta = self.add_variable('beta',[shape],initializer=tf.initializers.constant(values[3]),trainable=True)

	def update(self,variable,value):
		delta = (variable - value) * self.decay
		variable.assign_sub(delta)

	def _write_caffe(self, btm):
		global caffe_string, layer_counter
		if not 'caffe_string' in globals():
			caffe_string = ''
		if not 'layer_counter' in globals():
			layer_counter = 0

		layer_name = 'bn%d'%layer_counter

		caffe_string += 'layer{\n'
		caffe_string += ' name: "%s"\n'%layer_name
		caffe_string += ' type: "BatchNorm"\n'
		caffe_string += ' bottom: "%s"\n'%btm()
		caffe_string += ' top: "%s"\n'%btm()
		caffe_string += ' batch_norm_param{\n  use_global_stats:true\n  eps:1e-5\n }\n'
		caffe_string += '}\n'

		params_dict[layer_name] = {}
		params_dict[layer_name]['mean'] = self.moving_average.numpy()
		params_dict[layer_name]['var'] = self.variance.numpy()
		params_dict[layer_name]['scale'] = 1.

		layer_name = 'scale%d'%layer_counter

		caffe_string += 'layer{\n'
		caffe_string += ' name: "%s"\n'%layer_name
		caffe_string += ' type: "Scale"\n'
		caffe_string += ' bottom: "%s"\n'%btm()
		caffe_string += ' top: "%s"\n'%btm()
		caffe_string += ' scale_param{\n  bias_term:true\n }\n'
		caffe_string += '}\n'
		params_dict[layer_name] = {}
		params_dict[layer_name]['scale'] = self.gamma.numpy()
		params_dict[layer_name]['bias'] = self.beta.numpy()
		return btm

	def call(self, x):
		"""
		:param x: Input tensor or numpy array. The object will be automatically converted to tensor if the input is np.array. Note that other arrays in args or kwargs will not be auto-converted.
		"""
		name = x[1] 
		x = x[0]
		if self.is_training is None:
			is_training = bool(tf.keras.backend.learning_phase())
		else:
			is_training = self.is_training
		# is_training = True
		# print(is_training, time.time())
		inp_shape = x.get_shape().as_list()
		inp_dim_num = len(inp_shape)
		if inp_dim_num==3:
			x = tf.expand_dims(x, axis=1)
		elif inp_dim_num==2:
			x = tf.expand_dims(x, axis=1)
			x = tf.expand_dims(x, axis=1)
		elif inp_dim_num==5:
			x = tf.reshape(x, [inp_shape[0], inp_shape[1], inp_shape[2]*inp_shape[3], inp_shape[4]])
		if is_training:
			res, mean, var = tf.compat.v1.nn.fused_batch_norm(x, self.gamma, self.beta, None, None, self.epsilon, is_training=is_training)
			self.update(self.moving_average, mean)
			self.update(self.variance, var)
		else:
			res, mean, var = tf.compat.v1.nn.fused_batch_norm(x, self.gamma, self.beta, self.moving_average, self.variance, self.epsilon, is_training=is_training)
		if inp_dim_num==3:
			res = tf.squeeze(res , axis=1)
		elif inp_dim_num==2:
			res = tf.squeeze(res, axis=[1,2])
		elif inp_dim_num==5:
			res = tf.reshape(res, inp_shape)
		lname = self._write_caffe(name)
		return res, lname

class flatten(KLayer):
	"""
	Basic flatten layer
	"""
	def __init__(self):
		super(flatten, self).__init__()

	def build(self, input_shape):
		self.shape = input_shape

	def _write_caffe(self, name):
		global caffe_string, layer_counter
		if not 'caffe_string' in globals():
			caffe_string = ''
		if not 'layer_counter' in globals():
			layer_counter = 0
		layer_name = 'flatten%d'%layer_counter

		caffe_string += 'layer{\n'
		caffe_string += ' name: "%s"\n'%layer_name
		caffe_string += ' type: "Flatten"\n'
		caffe_string += ' bottom: "%s"\n'%name()
		caffe_string += ' top: "%s"\n'%layer_name
		# caffe_string += ' crop_param{\n  offset:%d\n  offset:%d\n  }\n}\n'%(1,1)
		caffe_string += '}\n'

		layer_counter += 1 
		return helper.LayerName(layer_name)

	def call(self, x):
		"""
		:param x: Input tensor or numpy array. The object will be automatically converted to tensor if the input is np.array. Note that other arrays in args or kwargs will not be auto-converted.
		"""
		name = x[1]
		x = x[0]
		self.shape = x.get_shape().as_list()
		num = 1
		for k in self.shape[1:]:
			num *= k 
		res = tf.reshape(x, [-1, num])
		lname = self._write_caffe(name)
		return res , lname

class NNUpSample2D(KLayer):
	"""docstring for NNUpSample"""
	def __init__(self, factor):
		super(NNUpSample2D, self).__init__()
		self.factor = factor
	def _get_weights(self):
		w = np.zeros([self.factor, self.factor, self.chn, self.chn])
		w = np.float32(w)
		for i in range(self.chn):
			w[:,:,i,i] = 1
		return w 
	def build(self, input_shape):
		self.chn = input_shape[0][-1]
	def _write_caffe(self, name):
		global caffe_string, layer_counter
		if not 'caffe_string' in globals():
			caffe_string = ''
		if not 'layer_counter' in globals():
			layer_counter = 0
		layer_name = 'nnup%d'%layer_counter

		caffe_string += 'layer{\n'
		caffe_string += ' name: "%s"\n'%layer_name
		caffe_string += ' type: "Deconvolution"\n'
		caffe_string += ' bottom: "%s"\n'%name()
		caffe_string += ' top: "%s"\n'%layer_name
		caffe_string += ' convolution_param{\n'
		caffe_string += '  num_output: %d\n'%self.chn
		caffe_string += '  bias_term: %s\n'%('false')
		caffe_string += '  stride: %d\n'%self.factor
		caffe_string += '  kernel_h: %d\n'%(self.factor)
		caffe_string += '  kernel_w: %d\n'%(self.factor)
		caffe_string += ' }\n}\n'

		params_dict[layer_name] = {}
		params_dict[layer_name]['kernel'] = self._get_weights()

		layer_counter += 1 
		return helper.LayerName(layer_name)

	def call(self, x):
		name = x[1]
		x = x[0]
		shape = x.get_shape().as_list()
		w = self._get_weights()
		outshape = [shape[0], shape[1]*self.factor, shape[2]*self.factor, self.chn]
		stride = [1, self.factor, self.factor, 1]
		x = tf.nn.conv2d_transpose(x, w, outshape, stride)
		lname = self._write_caffe(name)
		return x, lname

class BroadcastMUL(KLayer):
	def __init__(self):
		super(BroadcastMUL, self).__init__()

	def _write_caffe(self, names, tiles, outchn):
		global caffe_string, layer_counter
		if not 'caffe_string' in globals():
			caffe_string = ''
		if not 'layer_counter' in globals():
			layer_counter = 0
		

		# manual tiling layers to match the size 
		# layer_name = 'tile_0_%d'%layer_counter
		# caffe_string += 'layer{\n'
		# caffe_string += ' name: "%s"\n'%layer_name
		# caffe_string += ' type: "Tile"\n'
		# caffe_string += ' bottom:"%s"\n'%names[0]()
		# caffe_string += ' top: "%s"\n'%layer_name
		# caffe_string += ' tile_param{\n  axis:2\n  tiles:%d\n }\n'%tiles
		# caffe_string += '}\n'

		# layer_name = 'tile_1_%d'%layer_counter
		# caffe_string += 'layer{\n'
		# caffe_string += ' name: "%s"\n'%layer_name
		# caffe_string += ' type: "Tile"\n'
		# caffe_string += ' bottom:"tile_0_%d"\n'%layer_counter
		# caffe_string += ' top: "%s"\n'%layer_name
		# caffe_string += ' tile_param{\n  axis:3\n  tiles:%d\n }\n'%tiles
		# caffe_string += '}\n'

		layer_name = 'tile_0_%d'%layer_counter

		caffe_string += 'layer{\n'
		caffe_string += ' name: "%s"\n'%layer_name
		caffe_string += ' type: "Deconvolution"\n'
		caffe_string += ' bottom: "%s"\n'%names[0]()
		caffe_string += ' top: "%s"\n'%layer_name
		caffe_string += ' convolution_param{\n'
		caffe_string += '  num_output: %d\n'%outchn
		caffe_string += '  bias_term: %s\n'%('false')
		caffe_string += '  group: %d\n'%outchn
		caffe_string += '  stride: 1\n'
		caffe_string += '  pad_h: 0\n'
		caffe_string += '  pad_w: 0\n'
		caffe_string += '  kernel_h: %d\n'%tiles
		caffe_string += '  kernel_w: %d\n'%tiles
		caffe_string += ' }\n}\n'

		params_dict[layer_name] = {}
		params_dict[layer_name]['dwkernel'] = np.ones([tiles, tiles, outchn, 1]).astype(np.float32)

		# do multiplication
		layer_name = 'mul%d'%layer_counter
		caffe_string += 'layer{\n'
		caffe_string += ' name: "%s"\n'%layer_name
		caffe_string += ' type: "Eltwise"\n'
		caffe_string += ' bottom:"tile_0_%d"\n'%layer_counter
		caffe_string += ' bottom:"%s"\n'%names[1]()
		caffe_string += ' top: "%s"\n'%layer_name
		caffe_string += ' eltwise_param{\n  operation:PROD\n }\n'
		caffe_string += '}\n'
		layer_counter += 1
		return helper.LayerName(layer_name)

	def call(self, x):
		names = [i[1] for i in x]
		xs = [i[0] for i in x]
		out = xs[0]*xs[1]
		lname = self._write_caffe(names, xs[1].shape[1], xs[1].shape[-1])
		return out, lname

class SUM(KLayer):
	def __init__(self):
		super(SUM, self).__init__()
	def _write_caffe(self, names):
		global caffe_string, layer_counter
		if not 'caffe_string' in globals():
			caffe_string = ''
		if not 'layer_counter' in globals():
			layer_counter = 0
		layer_name = 'add%d'%layer_counter

		caffe_string += 'layer{\n'
		caffe_string += ' name: "%s"\n'%layer_name
		caffe_string += ' type: "Eltwise"\n'
		for n in names:
			caffe_string += ' bottom:"%s"\n'%n()
		caffe_string += ' top: "%s"\n'%layer_name
		caffe_string += ' eltwise_param{\n  operation:SUM\n }\n'
		caffe_string += '}\n'
		layer_counter += 1
		return helper.LayerName(layer_name)
	def call(self, x):
		names = [i[1] for i in x]
		xs = [i[0] for i in x]
		lname = self._write_caffe(names)
		return sum(xs), lname

class CONCAT(KLayer):
	def __init__(self):
		super(CONCAT, self).__init__()
	def _write_caffe(self, names):
		global caffe_string, layer_counter
		if not 'caffe_string' in globals():
			caffe_string = ''
		if not 'layer_counter' in globals():
			layer_counter = 0
		layer_name = 'concat%d'%layer_counter

		caffe_string += 'layer{\n'
		caffe_string += ' name: "%s"\n'%layer_name
		caffe_string += ' type: "Concat"\n'
		for n in names:
			caffe_string += ' bottom:"%s"\n'%n()
		caffe_string += ' top: "%s"\n'%layer_name
		caffe_string += ' concat_param{\n  axis:1\n }\n'
		caffe_string += '}\n'
		layer_counter += 1
		return helper.LayerName(layer_name)
	def call(self, x):
		names = [i[1] for i in x]
		xs = [i[0] for i in x]
		lname = self._write_caffe(names)
		return tf.concat(xs, axis=-1), lname

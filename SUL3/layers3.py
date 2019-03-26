import tensorflow as tf 
from tensorflow.keras.layers import Layer 
import numpy as np 

class conv2D(Layer):
	def __init__(self, size, outchn, stride=1,pad='SAME',usebias=True,values=None,dilation_rate=1):
		super(conv2D, self).__init__()
		self.size = size
		self.outchn = outchn
		self.stride = stride
		self.pad = pad 
		self.usebias = usebias
		self.values = values
		self.dilation_rate = dilation_rate

	def _parse_args(self, input_shape):
		inchannel = input_shape[-1]
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
		self._parse_args(input_shape)
		if self.values is not None:
			self.kernel = self.add_variable('kernel', shape=self.size, initializer=tf.initializers.constant(values[0]))
		else:
			self.kernel = self.add_variable('kernel', shape=self.size, initializer=tf.initializers.GlorotUniform())
		if self.usebias:
			if self.values is not None:
				self.bias = self.add_variable('bias', shape=[self.outchn], initializer=tf.initializers.constant(values[1]))
			else:
				self.bias = self.add_variable('bias', shape=[self.outchn], initializer=tf.initializers.constant(0.0))

	def call(self, x):
		out = tf.nn.conv2d(x, self.kernel, self.stride, self.pad, dilations=self.dilation_rate)
		if self.usebias:
			out = tf.nn.bias_add(out, self.bias)
		return out 

class conv3D(Layer):
	def __init__(self,size,outchn,stride=1,pad='SAME',usebias=True,values=None,dilation_rate=1):
		super(conv3D, self).__init__()
		self.size = size
		self.outchn = outchn
		self.stride = stride
		self.pad = pad 
		self.usebias = usebias
		self.values = values
		self.dilation_rate = dilation_rate

	def _parse_args(self, input_shape):
		# set size
		inchannel = input_shape[-1]
		if isinstance(self.size,list):
			self.size = [self.size[0], self.size[1], self.size[2],inchannel,self.outchn]
		else:
			self.size = [self.size, self.size, self.size, inchannel, self.outchn]
		# set stride
		if isinstance(self.stride,list):
			self.stride = [1,self.stride[0],self.stride[1], self.stride[2],1]
		else:
			self.stride = [1,self.stride, self.stride, self.stride, 1]
		# set dilation
		if isinstance(self.dilation_rate,list):
			self.dilation_rate = [1,self.dilation_rate[0],self.dilation_rate[1],self.dilation_rate[2],1]
		else:
			self.dilation_rate = [1,self.dilation_rate,self.dilation_rate,self.dilation_rate,1]

	def build(self, input_shape):
		self._parse_args(input_shape)
		if self.values is not None:
			self.kernel = self.add_variable('kernel', shape=self.size, initializer=tf.initializers.constant(values[0]))
		else:
			self.kernel = self.add_variable('kernel', shape=self.size, initializer=tf.initializers.GlorotUniform())
		if self.usebias:
			if self.values is not None:
				self.bias = self.add_variable('bias', shape=[self.outchn], initializer=tf.initializers.constant(values[1]))
			else:
				self.bias = self.add_variable('bias', shape=[self.outchn], initializer=tf.initializers.constant(0.0))
		
	def call(self,x):
		out = tf.nn.conv3d(x,self.kernel,self.stride,self.pad,dilations=self.dilation_rate)
		if self.usebias:
			out = tf.nn.bias_add(out,self.bias)
		return out 

class conv1D(Layer):
	def __init__(self,size,outchn,stride=1,pad='SAME',usebias=True,values=None,dilation_rate=1):
		super(conv1D, self).__init__()
		self.size = size
		self.outchn = outchn
		self.stride = stride
		self.pad = pad 
		self.usebias = usebias
		self.values = values
		self.dilation_rate = dilation_rate

	def _parse_args(self, input_shape):
		# set size
		inchannel = input_shape[-1]
		self.size = [self.size, inchannel, self.outchn]
		# set stride
		self.stride = [1, self.stride, 1]
		# set dilation
		self.dilation_rate = [1,self.dilation_rate,1]

	def build(self, input_shape):
		self._parse_args(input_shape)
		if self.values is not None:
			self.kernel = self.add_variable('kernel', shape=self.size, initializer=tf.initializers.constant(values[0]))
		else:
			self.kernel = self.add_variable('kernel', shape=self.size, initializer=tf.initializers.GlorotUniform())
		if self.usebias:
			if self.values is not None:
				self.bias = self.add_variable('bias', shape=[self.outchn], initializer=tf.initializers.constant(values[1]))
			else:
				self.bias = self.add_variable('bias', shape=[self.outchn], initializer=tf.initializers.constant(0.0))
		
	def call(self, x):
		# x = tf.expand_dims(x, axis=1)
		out = tf.nn.conv1d(x,self.kernel,self.stride,self.pad,dilations=self.dilation_rate)
		if self.usebias:
			out = tf.nn.bias_add(out,self.bias)
		# out = tf.squeeze(out, axis=1)
		return out 

class deconv1D(Layer):
	def __init__(self,size,outchn,stride=1,pad='SAME',usebias=True,values=None,dilation_rate=1):
		super(conv1D, self).__init__()
		self.size = size
		self.outchn = outchn
		self.stride = stride
		self.pad = pad 
		self.usebias = usebias
		self.values = values
		self.dilation_rate = dilation_rate

	def _parse_args(self, input_shape):
		# set size
		inchannel = input_shape[-1]
		self.size = [self.size, inchannel, self.outchn]
		# set stride
		self.stride = [1, self.stride, 1]
		# set dilation
		self.dilation_rate = [1,self.dilation_rate,1]
		# compute output shape
		if self.pad=='SAME':
			self.outshape = [input_shape[0], input_shape[1]*self.stride[1], input_shape[2]]
		else:
			self.outshape = [input_shape[0], input_shape[1]*self.stride[1]+self.size[1]-self.stride[1], inp_shape[2]]

	def build(self, input_shape):
		self._parse_args(input_shape)
		if self.values is not None:
			self.kernel = self.add_variable('kernel', shape=self.size, initializer=tf.initializers.constant(values[0]))
		else:
			self.kernel = self.add_variable('kernel', shape=self.size, initializer=tf.initializers.GlorotUniform())
		if self.usebias:
			if self.values is not None:
				self.bias = self.add_variable('bias', shape=[self.outchn], initializer=tf.initializers.constant(values[1]))
			else:
				self.bias = self.add_variable('bias', shape=[self.outchn], initializer=tf.initializers.constant(0.0))
		
	def call(self, x):
		# x = tf.expand_dims(x, axis=1)
		out = tf.nn.conv1d_transpose(x,self.kernel,self.stride,self.pad,dilations=self.dilation_rate)
		if self.usebias:
			out = tf.nn.bias_add(out,self.bias)
		# out = tf.squeeze(out, axis=1)
		return out 

class deconv2D(Layer):
	def __init__(self,size,outchn,stride=1,pad='SAME',usebias=True,values=None,dilation_rate=1):
		super(conv1D, self).__init__()
		self.size = size
		self.outchn = outchn
		self.stride = stride
		self.pad = pad 
		self.usebias = usebias
		self.values = values
		self.dilation_rate = dilation_rate

	def _parse_args(self, input_shape):
		# set size
		inchannel = input_shape[-1]
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

		# compute output shape
		if self.pad=='SAME':
			self.outshape = [input_shape[0], input_shape[1]*self.stride[1], input_shape[2]*self.stride[2], input_shape[3]]
		else:
			self.outshape = [input_shape[0], input_shape[1]*self.stride[1]+self.size[1]-self.stride[1], inp_shape[2]*self.stride[2]+self.size[2]-self.stride[2], input_shape[3]]

	def build(self, input_shape):
		self._parse_args(input_shape)
		if self.values is not None:
			self.kernel = self.add_variable('kernel', shape=self.size, initializer=tf.initializers.constant(values[0]))
		else:
			self.kernel = self.add_variable('kernel', shape=self.size, initializer=tf.initializers.GlorotUniform())
		if self.usebias:
			if self.values is not None:
				self.bias = self.add_variable('bias', shape=[self.outchn], initializer=tf.initializers.constant(values[1]))
			else:
				self.bias = self.add_variable('bias', shape=[self.outchn], initializer=tf.initializers.constant(0.0))
		
	def call(self, x):
		# x = tf.expand_dims(x, axis=1)
		out = tf.nn.conv2d_transpose(x,self.kernel,self.stride,self.pad,dilations=self.dilation_rate)
		if self.usebias:
			out = tf.nn.bias_add(out,self.bias)
		# out = tf.squeeze(out, axis=1)
		return out 

class deconv3D(Layer):
	def __init__(self,size,outchn,stride=1,pad='SAME',usebias=True,values=None,dilation_rate=1):
		super(conv1D, self).__init__()
		self.size = size
		self.outchn = outchn
		self.stride = stride
		self.pad = pad 
		self.usebias = usebias
		self.values = values
		self.dilation_rate = dilation_rate

	def _parse_args(self, input_shape):
		# set size
		inchannel = input_shape[-1]
		if isinstance(self.size,list):
			self.size = [self.size[0], self.size[1], self.size[2],inchannel,self.outchn]
		else:
			self.size = [self.size, self.size, self.size, inchannel, self.outchn]
		# set stride
		if isinstance(self.stride,list):
			self.stride = [1,self.stride[0],self.stride[1], self.stride[2],1]
		else:
			self.stride = [1,self.stride, self.stride, self.stride, 1]
		# set dilation
		if isinstance(self.dilation_rate,list):
			self.dilation_rate = [1,self.dilation_rate[0],self.dilation_rate[1],self.dilation_rate[2],1]
		else:
			self.dilation_rate = [1,self.dilation_rate,self.dilation_rate,self.dilation_rate,1]
		# compute output shape
		if self.pad=='SAME':
			self.outshape = [input_shape[0], input_shape[1]*self.stride[1], input_shape[2]*self.stride[2], input_shape[3]*self.stride[3], input_shape[4]]
		else:
			self.outshape = [input_shape[0], input_shape[1]*self.stride[1]+self.size[1]-self.stride[1], input_shape[2]*self.stride[2]+self.size[2]-self.stride[2], input_shape[3]*self.stride[3]+self.size[3]-self.stride[3], input_shape[4]]

	def build(self, input_shape):
		self._parse_args(input_shape)
		if self.values is not None:
			self.kernel = self.add_variable('kernel', shape=self.size, initializer=tf.initializers.constant(values[0]))
		else:
			self.kernel = self.add_variable('kernel', shape=self.size, initializer=tf.initializers.GlorotUniform())
		if self.usebias:
			if self.values is not None:
				self.bias = self.add_variable('bias', shape=[self.outchn], initializer=tf.initializers.constant(values[1]))
			else:
				self.bias = self.add_variable('bias', shape=[self.outchn], initializer=tf.initializers.constant(0.0))
		
	def call(self, x):
		# x = tf.expand_dims(x, axis=1)
		out = tf.nn.conv3d_transpose(x,self.kernel,self.stride,self.pad,dilations=self.dilation_rate)
		if self.usebias:
			out = tf.nn.bias_add(out,self.bias)
		# out = tf.squeeze(out, axis=1)
		return out 

class maxpoolLayer(Layer):
	def __init__(self, size, stride, pad='SAME'):
		super(maxpoolLayer, self).__init__()
		self.size = size
		self.stride = stride
		self.pad = pad 

	def call(self, x):
		out = tf.nn.max_pool(x, self.size, self.stride, self.pad)
		return out 

class avgpoolLayer(Layer):
	def __init__(self, size, stride, pad='SAME'):
		super(avgpoolLayer, self).__init__()
		self.size = size 
		self.stride = stride
		self.pad = pad 

	def call(self, x):
		out = tf.nn.avg_pool(x, self.size, self.stride, self.pad)
		return out 

class activation(Layer):
	def __init__(self, param, **kwargs):
		super(activation, self).__init__()

		self.param = param
		self.kwargs = kwargs

	def call(self, x):
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
		else:
			res =  x
		return res

class fcLayer(Layer):
	def __init__(self, outsize, usebias=True, values=None):
		super(fcLayer, self).__init__()
		self.outsize = outsize
		self.usebias = usebias
		self.values = values

	def _parse_args(self, input_shape):
		# set size
		insize = input_shape[-1]
		self.size = [insize, self.outsize]

	def build(self, input_shape):
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

	def call(self, x):
		res = tf.matmul(x, self.kernel)
		if self.usebias:
			res = tf.nn.bias_add(res, self.bias)
		return res 

class batch_norm(Layer):
	def __init__(self, decay=0.01, epsilon=0.001, is_training=None, values=None):
		super(batch_norm, self).__init__()

		self.decay = decay
		self.epsilon = epsilon
		self.is_training = is_training
		self.values = values

	def build(self, input_shape):
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

	def call(self, x):
		if self.is_training is None:
			is_training = tf.keras.backend.learning_phase()
		else:
			is_training = self.is_training
		inp_shape = x.get_shape().as_list()
		inp_dim_num = len(inp_shape)
		if inp_dim_num==3:
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
		elif inp_dim_num==5:
			res = tf.reshape(res, inp_shape)
		return res 

class flatten(Layer):
	def __init__(self):
		super(flatten, self).__init__()

	def build(self, input_shape):
		self.shape = input_shape

	def call(self, x):
		num = 1
		for k in self.shape[1:]:
			num *= k 
		res = tf.reshape(x, [-1, num])
		return res 

class graphConvLayer(Layer):
	def __init__(self, outsize, adj_mtx=None, adj_fn=None, values=None, usebias=True):
		super(graphConvLayer, self).__init__()
		assert (adj_mtx is None) ^ (adj_fn is None), 'Assign either adj_mtx or adj_fn' 
		self.outsize = outsize
		self.adj_mtx = adj_mtx
		self.adj_fn = adj_fn
		self.values = values
		self.usebias = usebias
		self.normalized = False

	def _parse_args(self, input_shape):
		# set size
		insize = input_shape[-1]
		self.size = [insize, self.outsize]

	def build(self, input_shape):
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

	def _normalize_adj_mtx(self, mtx):
		S = tf.reduce_sum(mtx, axis=1)
		S = tf.sqrt(S)
		S = 1. / S
		S = tf.diag(S)
		I = tf.eye(tf.cast(S.shape[0], tf.int64))
		A_ = (mtx + I) 
		A_ = tf.matmul(S, A_)
		A_ = tf.matmul(A_, S)
		return tf.stop_gradient(A_)

	def call(self, x):
		if self.adj_mtx is not None:
			A = self.adj_mtx
			if not self.normalized:
				A = self._normalize_adj_mtx(A)
				self.adj_mtx = A 
				self.normalized = True
		else:
			A = self.adj_fn(x)
			A = self._normalize_adj_mtx(A)

		res = tf.matmul(A, x)
		res = tf.matmul(res, self.W)
		if self.usebias:
			res = tf.nn.bias_add(res, self.b)
		return res 

class bilinearUpSample(Layer):
	def __init__(self, factor):
		self.factor = factor
		super(bilinearUpSample, self).__init__()

	def upsample_kernel(size):
		factor = (size +1)//2
		if size%2==1:
			center = factor - 1
		else:
			center = factor - 0.5
		og = np.ogrid[:size, :size]
		return (1 - abs(og[0]-center)/factor) * (1-abs(og[1]-center)/factor)

	def upsample_kernel_1d(size):
		factor = (size + 1)//2
		if size%2==1:
			center = factor - 1
		else:
			center = factor - 0.5
		og = np.ogrid[:size]
		og = og[None, :]
		kernel = 1 - abs(og - center)/factor
		return kernel

	def upsample_kernel_3d(size):
		factor = (size + 1)//2
		if size%2==1:
			center = factor - 1
		else:
			center = factor - 0.5
		og = np.ogrid[:size, :size, :size]
		kernel = (1 - abs(og[0]-center)/factor) * (1-abs(og[1]-center)/factor) * (1-abs(og[2]-center)/factor)
		return kernel

	def get_kernel(self, dim, chn, factor):
		filter_size = 2*factor - factor%2
		shape = [filter_size] * (dim-2) + [num_featuremap, num_featuremap]
		weights = np.zeros(shape, dtype=np.float32)
		if dim == 5:
			k = upsample_kernel_3d(filter_size)
			for i in range(chn):
				weights[:,:,:,i,i] = kernel
		elif dim==4:
			k = upsample_kernel(filter_size)
			for i in range(chn):
				weights[:,:,i,i] = kernel
		elif dim==3:
			k = upsample_kernel_1d(filter_size)
			for i in range(chn):
				weights[:,i,i] = kernel
		return weights

	def build(self, input_shape):
		self.dim = len(input_shape)
		self.num_chn = input_shape[-1]
		if self.dim==3:
			self.outshape = [input_shape[0], input_shape[1]*self.factor, input_shape[2]]
		elif self.dim==4:
			self.outshape = [input_shape[0], input_shape[1]*self.factor, input_shape[2]*self.factor, input_shape[3]]
		elif self.dim==5:
			self.outshape = [input_shape[0], input_shape[1]*self.factor, input_shape[2]*self.factor, input_shape[3]*self.factor, input_shape[4]]
		self.stride = [1] + [self.factor]*(self.dim-2) + [1]
		kernel = self.get_kernel(self.dim, self.chn, self.factor)
		self.kernel = self.add_variable('kernel_upsample', shape=kernel.shape, initializer=tf.initializers.constant(kernel))

	def call(self, x):
		pad = [[0,0]] + [[1,1]]*(self.dim-2) + [[0,0]]
		x = tf.pad(x, pad, mode='symmetric')

		if self.dim==3:
			res = tf.nn.conv1d_transpose(x, self.kernel, self.outshape, self.stride)
			res = res[:, factor:-factor, :]
		elif self.dim==4:
			res = tf.nn.conv2d_transpose(x, self.kernel, self.outshape, self.stride)
			res = res[:, factor:-factor, factor:-factor, :]
		elif self.dim==5:
			res = tf.nn.conv3d_transpose(x, self.kernel, self.outshape, self.stride)
			res = res[:, factor:-factor, factor:-factor, factor:-factor, :]
		return res 

@tf.custom_gradient
def gradient_reverse(x):
	def grad(dy):
		return -dy 
	return x, grad

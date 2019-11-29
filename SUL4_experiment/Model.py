from . import Layers as L 
import tensorflow as tf 
import numpy as np 
import os 
from tensorflow.keras import Model as KModel
from tensorflow.python.training.tracking.data_structures import NoDependency

# activation const
PARAM_RELU = 0
PARAM_LRELU = 1
PARAM_ELU = 2
PARAM_TANH = 3
PARAM_MFM = 4
PARAM_MFM_FC = 5
PARAM_SIGMOID = 6
PARAM_SWISH = 7

######## util functions ###########
def accuracy(pred,y,name='acc', one_hot=True):
	if not one_hot:
		correct = tf.equal(tf.cast(tf.argmax(pred,-1),tf.int64),tf.cast(tf.argmax(y,-1),tf.int64))
	else:
		correct = tf.equal(tf.cast(tf.argmax(pred,-1),tf.int64),tf.cast(y,tf.int64))
	acc = tf.reduce_mean(tf.cast(correct,tf.float32))
	return acc

def weight_decay(wd, model):
	w_reg = wd * 0.5 * sum([tf.reduce_sum(tf.square(w)) for w in model.trainable_variables]) 
	return w_reg

def weight_decay_v2(wd, model):
	ws = [w for w in model.trainable_variables if 'kernel' in w.name]
	w_reg = wd * 0.5 * sum([tf.reduce_sum(tf.square(w)) for w in ws]) 
	return w_reg

def spectrum_regularizer(kernel, cutoff_ratio=0.5, size=50, filter_type='high'):
	assert filter_type in ['high','low']
	s = tf.signal.rfft2d(k, fft_length=[size,size])
	s = tf.abs(s)
	mask = np.float32(np.ones([size,size//2+1]))
	r = int(size*cutoff_ratio)
	mask[-r:,:r] = 0
	mask[:r, :r] = 0
	if filter_type=='low':
		mask = 1 - mask
	mask = tf.convert_to_tensor(mask)
	s = s*mask 
	return s 

################
# dumb model declaration
# make alias for init and call
class Model(KModel):
	"""
	Model template
	"""
	def __init__(self, *args, **kwargs):
		"""
		The default initialization. Not recommended to touch.
		"""
		super(Model, self).__init__()
		self.graph_initialized = False
		self.initialize(*args, **kwargs)

	def initialize(self, *args, **kwargs):
		"""
		Initialization function. Set layers, sub-modules and parameters here.
		"""
		pass 

	def call(self, x, *args, **kwargs):
		"""
		Default function for class callables. Not recommended to touch.
		"""
		self.graph_initialized = True
		if 'training' in kwargs:
			kwargs.pop('training')
		return self.forward(x, *args, **kwargs)

	def forward(self, x, *args, **kwargs):
		"""
		Here is where the model logits locates.

		:param x: Input tensor or numpy array. The object will be automatically converted to tensor if the input is np.array. Note that other arrays in args or kwargs will not be auto-converted.
		"""
		pass 

################
# Layer Class 

class ConvLayer(KModel):
	"""
	High-level convolution 2D layer
	"""
	def __init__(self, size, outchn, dilation_rate=1, stride=1, pad='SAME', activation=-1, batch_norm=False, usebias=True, values=None):
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

		:param activation: Same candidates as layers3.activate

		:type batch_norm: bool
		:param batch_norm: Whether to use batch normalization in this layer.
		"""
		super(ConvLayer, self).__init__()
		self.batch_norm = batch_norm
		self.activation = activation
		self.values = values

		if values is None:
			self.conv = L.conv2D(size, outchn, stride, pad, dilation_rate, usebias)
			if batch_norm:
				self.bn = L.batch_norm()
		else:
			if usebias:
				idx = 2 
			else:
				idx = 1
			self.conv = L.conv2D(size, outchn, stride, pad, dilation_rate, usebias, values=values[:idx])
			if batch_norm:
				self.bn = L.batch_norm(values=values[idx:])
		if activation!=-1:
			self.act = L.activation(activation)

	def call(self, x):
		"""
		:param x: Input tensor or numpy array. The object will be automatically converted to tensor if the input is np.array. Note that other arrays in args or kwargs will not be auto-converted.
		
		:return: Tensor or a list of tensor.
		"""
		x = self.conv(x)
		if self.batch_norm:
			x = self.bn(x)
		if self.activation!=-1:
			x = self.act(x)
		return x 

class ConvLayer3D(KModel):
	"""
	High-level convolution 3D layer
	"""
	def __init__(self, size, outchn, dilation_rate=1, stride=1, pad='SAME', activation=-1, batch_norm=False, usebias=True, values=None):
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

		:param activation: Same candidates as layers3.activate

		:type batch_norm: bool
		:param batch_norm: Whether to use batch normalization in this layer.
		"""
		super(ConvLayer3D, self).__init__()
		self.batch_norm = batch_norm
		self.activation = activation
		self.values = values

		if values is None:
			self.conv = L.conv3D(size, outchn, stride, pad, dilation_rate, usebias)
			if batch_norm:
				self.bn = L.batch_norm()
		else:
			if usebias:
				idx = 2
			else:
				idx = 1
			self.conv = L.conv3D(size, outchn, stride, pad, dilation_rate, usebias, values=values[:idx])
			if batch_norm:
				self.bn = L.batch_norm(values=values[idx:])
		if activation!=-1:
			self.act = L.activation(activation)

	def call(self, x):
		"""
		:param x: Input tensor or numpy array. The object will be automatically converted to tensor if the input is np.array. Note that other arrays in args or kwargs will not be auto-converted.
		
		:return: Tensor or a list of tensor.
		"""
		x = self.conv(x)
		if self.batch_norm:
			x = self.bn(x)
		if self.activation!=-1:
			x = self.act(x)
		return x 

class ConvLayer1D(KModel):
	"""
	High-level convolution 1D layer
	"""
	def __init__(self, size, outchn, dilation_rate=1, stride=1, pad='SAME', activation=-1, batch_norm=False, usebias=True):
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

		:param activation: Same candidates as layers3.activate

		:type batch_norm: bool
		:param batch_norm: Whether to use batch normalization in this layer.
		"""
		super(ConvLayer1D, self).__init__()
		self.conv = L.conv1D(size, outchn, stride, pad, dilation_rate, usebias)
		self.batch_norm = batch_norm
		self.activation = activation
		if batch_norm:
			self.bn = L.batch_norm()
		if activation!=-1:
			self.act = L.activation(activation)

	def call(self, x):
		"""
		:param x: Input tensor or numpy array. The object will be automatically converted to tensor if the input is np.array. Note that other arrays in args or kwargs will not be auto-converted.
		
		:return: Tensor or a list of tensor.
		"""
		x = self.conv(x)
		if self.batch_norm:
			x = self.bn(x)
		if self.activation!=-1:
			x = self.act(x)
		return x 

class DWConvLayer(KModel):
	"""
	High-level depth-wise convolution 2D layer
	"""
	def __init__(self, size, multiplier, dilation_rate=1, stride=1, pad='SAME', activation=-1, batch_norm=False, usebias=True, values=None):
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

		:param activation: Same candidates as layers3.activate

		:type batch_norm: bool
		:param batch_norm: Whether to use batch normalization in this layer.
		"""
		super(DWConvLayer, self).__init__()
		self.batch_norm = batch_norm
		self.activation = activation
		self.values = values

		if values is None:
			self.dwconv = L.dwconv2D(size, multiplier, stride, pad, dilation_rate, usebias)
			if batch_norm:
				self.bn = L.batch_norm()
		else:
			if usebias:
				idx = 2 
			else:
				idx = 1
			self.dwconv = L.dwconv2D(size, multiplier, stride, pad, dilation_rate, usebias, values=values[:idx])
			if batch_norm:
				self.bn = L.batch_norm(values=values[idx:])
		if activation!=-1:
			self.act = L.activation(activation)

	def call(self, x):
		"""
		:param x: Input tensor or numpy array. The object will be automatically converted to tensor if the input is np.array. Note that other arrays in args or kwargs will not be auto-converted.
		
		:return: Tensor or a list of tensor.
		"""
		x = self.dwconv(x)
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

class DeconvLayer(KModel):
	def __init__(self, size, outchn, dilation_rate=1, stride=1, pad='SAME', activation=-1, batch_norm=False, usebias=True):
		super(DeconvLayer, self).__init__()
		self.conv = L.deconv2D(size, outchn, stride, pad, dilation_rate, usebias)
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

class DeconvLayer3D(KModel):
	def __init__(self, size, outchn, dilation_rate=1, stride=1, pad='SAME', activation=-1, batch_norm=False, usebias=True):
		super(DeconvLayer3D, self).__init__()
		self.conv = L.deconv3D(size, outchn, stride, pad, dilation_rate, usebias)
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

class Dense(KModel):
	def __init__(self, outsize, batch_norm=False, activation=-1 , usebias=True, norm=False):
		super(Dense, self).__init__()
		self.fc = L.fcLayer(outsize, usebias, norm=norm)
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

class OctConv(Model):
	def initialize(self, size, chn, ratio, input_ratio=None, stride=1, pad='SAME', activation=-1, batch_norm=False, usebias=True):
		chn_big = int(chn*ratio)
		chn_small = chn - chn_big
		self.avgpool = AvgPool(2,2)
		self.Convhh = ConvLayer(size, chn_big, stride=stride, pad=pad, usebias=usebias)
		self.Convhl = ConvLayer(size, chn_small, stride=stride, pad=pad, usebias=usebias)
		self.Convlh = ConvLayer(size, chn_big, stride=stride, pad=pad, usebias=usebias)
		self.Convll = ConvLayer(size, chn_small, stride=stride, pad=pad, usebias=usebias)

		# bn and act
		if batch_norm:
			self.bn = L.batch_norm()
		if activation!=-1:
			self.act = L.activation(activation)

		self.batch_norm = batch_norm
		self.activation = activation
		self.ratio = ratio
		self.input_ratio = ratio if input_ratio is None else input_ratio
		self.stride = stride
		self.chn_big = chn_big
	
	def build(self, input_shape):
		self.imgsize = int(input_shape[1])
		chn = input_shape[-1]
		self.chn = chn 
		self.chn_inp_big = int(chn * self.input_ratio / (1 + self.input_ratio*3))
		self.chn_inp_small = chn - self.chn_inp_big

	def forward(self, x):
		big = x[:,:,:,:self.chn_inp_big*4]
		small = x[:,:,:,self.chn_inp_big*4:]
		big = tf.reshape(big, [-1, self.imgsize*2, self.imgsize*2, self.chn_inp_big])

		hh = self.Convhh(big)
		ll = self.Convll(small)

		hl = self.avgpool(big)
		hl = self.Convhl(hl)
		
		lh = self.Convlh(small)
		lh = tf.image.resize(lh, [self.imgsize*2//self.stride, self.imgsize*2//self.stride], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

		h_out = hh + lh
		h_out = tf.reshape(h_out, [-1, self.imgsize//self.stride, self.imgsize//self.stride, self.chn_big*4])
		l_out = ll + hl
		out = tf.concat([h_out, l_out], axis=-1)
		if self.batch_norm:
			out = self.bn(out)
		if self.activation!=-1:
			out = self.act(out)
		return out 

class OctMerge(Model):
	def initialize(self):
		self.avgpool = AvgPool(2,2)
	def build(self, input_shape):
		self.imgsize = int(input_shape[1])
		self.chn = int(input_shape[3])
	def forward(self,x):
		h = x 
		l = self.avgpool(x)
		h = tf.reshape(h, [-1, self.imgsize//2, self.imgsize//2, self.chn*4])
		out = tf.concat([h,l], axis=-1)
		return out 

class OctSplit(Model):
	def initialize(self, ratio):
		self.ratio = ratio
	def build(self, input_shape):
		self.imgsize = int(input_shape[1])
		chn = int(input_shape[-1])
		self.chn = chn
		self.chn_inp_big = int(chn * self.ratio / (1 + self.ratio*3))
		self.chn_inp_small = chn - self.chn_inp_big
	def forward(self, x):
		big = x[:,:,:,:self.chn_inp_big*4]
		small = x[:,:,:,self.chn_inp_big*4:]
		big = tf.reshape(big, [-1, self.imgsize*2, self.imgsize*2, self.chn_inp_big])
		big = tf.nn.space_to_depth(big, 2)
		res = tf.concat([big, small], axis=-1)
		return res 

class MarginalCosineLayer(Model):
	def initialize(self, num_classes):
		self.classifier = Dense(num_classes, usebias=False, norm=True)
	def forward(self, x, label, m1=1.0, m2=0.0, m3=0.0):
		# res = cos(m1t + m2) + m3
		# this loss will cause potential unstable
		label = tf.convert_to_tensor(label)
		x = tf.nn.l2_normalize(x, axis=1)
		x = self.classifier(x)
		if not(m1==1.0 and m2==0.0):
			t = tf.gather_nd(x, indices=tf.where(label>0.)) #shape: [N]
			if tf.keras.backend.floatx()=='float16':
				assert m1==1.0,'Only m1=1.0 is supported for fp16'
				sint = tf.sqrt(1.0 - tf.square(t))
				t = t * tf.cast(tf.cos(m2), tf.float16) - sint * tf.cast(tf.sin(m2), tf.float16)
			else:
				t = tf.math.acos(t)
				### original ###
				# if m1!=1.0:
				# 	t = t*m1
				# if m2!=0.0:
				# 	t = t+m2 
				### end ###
				### experimental: to limit the value not exceed pi ###
				if m1!=1.0:
					t = t*m1
					t1 = t * np.pi / tf.stop_gradient(t)
					t = tf.minimum(t,t1)
				if m2!=0.0:
					t = t+m2 
					t1 = t + np.pi - tf.stop_gradient(t)
					t = tf.minimum(t,t1)
				t = tf.math.cos(t)
			t = tf.expand_dims(t, axis=1)
			x = x*(1-label) + t*label
		x = x - label * m3
		return x

class QAttention(Model):
	def initialize(self, outdim):
		self.qe = L.fcLayer(outdim, usebias=False, norm=False)
	def forward(self, feature, query):
		query = tf.convert_to_tensor(query)
		def Qatt_batch(feat, q):
			q = tf.expand_dims(q,axis=0)
			e = tf.matmul(feat, q, transpose_b=True) # [featsize, 1]
			e = tf.nn.softmax(e)
			out = e * feat
			out = tf.reduce_mean(out,0)
			return out
		result =  tf.map_fn(lambda x:Qatt_batch(x[0],x[1]) , (feature, query), dtype=tf.float32)
		return result

class SelfAttention(Model):
	def initialize(self, att_num, outnum, residual=True):
		self.layerf = L.fcLayer(att_num)
		self.layerg = L.fcLayer(att_num)
		self.layerh = L.fcLayer(outnum)
		self.residual = residual

	def forward(self, x):
		f = self.layerf(x)
		g = self.layerg(x)
		h = self.layerh(x)
		att = tf.matmul(f, g, transpose_b=True)
		att = tf.nn.softmax(att, -1)
		out = tf.matmul(att, x)
		if self.residual:
			out = out + x
		return out 

class LSTMCell(Model):
	def initialize(self, outdim):
		self.F = L.fcLayer(outdim, usebias=False, norm=False)
		self.O = L.fcLayer(outdim, usebias=False, norm=False)
		self.I = L.fcLayer(outdim, usebias=False, norm=False)
		self.C = L.fcLayer(outdim, usebias=False, norm=False)

		self.hF = L.fcLayer(outdim, usebias=False, norm=False)
		self.hO = L.fcLayer(outdim, usebias=False, norm=False)
		self.hI = L.fcLayer(outdim, usebias=False, norm=False)
		self.hC = L.fcLayer(outdim, usebias=False, norm=False)

	def forward(self, x, h, c_prev):
		h = tf.convert_to_tensor(h)
		f = self.F(x) + self.hF(h)
		o = self.O(x) + self.hO(h)
		i = self.I(x) + self.hI(h)
		c = self.C(x) + self.hC(h)

		f_ = tf.math.sigmoid(f)
		c_ = tf.math.tanh(c) * tf.math.sigmoid(i)
		o_ = tf.math.sigmoid(o)

		next_c = c_prev * f_ + c_ 
		next_h = o_ * tf.math.tanh(next_c)
		return next_h, next_c

class LSTM(Model):
	def initialize(self, outdim):
		self.outdim = outdim
		self.LSTM = LSTMCell(outdim)
		self.h = None 
		self.c = None 
	def forward(self, x, init_hc=False):
		assert x[0] is not None, 'First element should not be None'
		outs = [] 
		if init_hc:
			self.h = tf.zeros([x[0].shape[0], self.outdim])
			self.c = tf.zeros([x[0].shape[0], self.outdim])
		for i in range(len(x)):
			next_inp = x[i]
			if next_inp is None:
				# next_inp = tf.stop_gradient(self.h)
				next_inp = self.h
			else:
				next_inp = tf.convert_to_tensor(next_inp)
			if self.h is None:
				self.h = tf.zeros([x[0].shape[0], self.outdim])
			if self.c is None:
				self.c = tf.zeros([x[0].shape[0], self.outdim])
			self.h, self.c = self.LSTM(next_inp, self.h, self.c)
			outs.append(self.h)
		return outs 

class ConvLSTM(Model):
	def initialize(self, chn):
		self.gx = M.ConvLayer(3, chn)
		self.gh = M.ConvLayer(3, chn)
		self.fx = M.ConvLayer(3, chn)
		self.fh = M.ConvLayer(3, chn)
		self.ox = M.ConvLayer(3, chn)
		self.oh = M.ConvLayer(3, chn)
		self.gx = M.ConvLayer(3, chn)
		self.gh = M.ConvLayer(3, chn)

	def forward(self, x, c, h):
		gx = self.gx(x)
		gh = self.gh(h)

		ox = self.ox(x)
		oh = self.oh(h)

		fx = self.fx(x)
		fh = self.fh(h)

		gx = self.gx(x)
		gh = self.gh(h)

		g = tf.tanh(gx + gh)
		o = tf.sigmoid(ox + oh)
		i = tf.sigmoid(ix + ih)
		f = tf.sigmoid(fx + fh)

		cell = f*c + i*g 
		h = o * tf.tanh(cell)
		return cell, h 

###############
# alias for layers
AvgPool = L.avgpoolLayer
MaxPool = L.maxpoolLayer
GlobalAvgPool = L.globalAvgpoolLayer
flatten = L.flatten()
BatchNorm = L.batch_norm
InstNorm = L.inst_norm
DeconvLayer2D = DeconvLayer
ConvLayer2D = ConvLayer
BilinearUpSample = L.bilinearUpSample
NALU = L.NALU

###############
# Saver 
class Saver():
	# DO NOT USE TF KERAS SAVE MODEL !!!!
	def __init__(self, model=None, optimizer=None):
		self.model = model 
		self.optimizer = optimizer
		if optimizer is None:
			self.checkpoint = tf.train.Checkpoint(model=model)
		else:
			self.checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

	def save(self, path):
		directory = os.path.dirname(path)
		if not os.path.exists(directory):
			os.makedirs(directory)
		self.checkpoint.save(path)
		print('Save to path:',path)

	def restore(self, path):
		try:
			if (not '.ckpt' in path):
				last_ckpt = tf.train.latest_checkpoint(path)
				if last_ckpt is None:
					print('No model found in checkpoint.')
					print('Model will auto-initialize after first iteration.')
					return 	
				else:
					path = last_ckpt
			self.checkpoint.restore(path)
			print('Model loaded:',path)
		except Exception as e :
			print(e)
			print('Model restore failed.')
			print('Model will auto-initialize after first iteration.')

###############
# accumulator
class GradAccumulator():
	def __init__(self):
		self.steps = 0
		self.grads = []

	def accumulate(self, grads):
		if len(grads)==0:
			self.grads = grads
		else:
			for old_g, new_g in zip(self.grads, grads):
				old_g.assign_add(new_g)
		self.steps += 1

	def get(self):
		res = [i/self.steps for i in self.grads]
		self.grads = []
		self.steps = 0
		return res 

	def get_step(self):
		return self.steps

###############
gradient_reverse = L.gradient_reverse

def pad(x, pad):
	if isinstance(pad, list):
		x = tf.pad(x, [[0,0],[pad[0],pad[1]], [pad[2],pad[3]], [0,0]])
	else:
		x = tf.pad(x, [[0,0],[pad,pad],[pad,pad],[0,0]])
	return x 

def pad1D(x, pad):
	if isinstance(pad, list):
		x = tf.pad(x, [[0,0],[pad[0],pad[1]], [0,0]])
	else:
		x = tf.pad(x, [[0,0],[pad,pad],[0,0]])
	return x 

def pad3D(x, pad):
	if isinstance(pad, list):
		x = tf.pad(x, [[0,0],[pad[0],pad[1]], [pad[2],pad[3]], [pad[4], pad[5]], [0,0]])
	else:
		x = tf.pad(x, [[0,0],[pad,pad],[pad,pad],[pad,pad],[0,0]])
	return x 

# not compatible to tf2
# use tf-addons
# def image_transform(x, H, out_shape=None, interpolation='BILINEAR'):
# 	# Will produce error if not specify 'output_shape' in eager mode
# 	shape = x.get_shape().as_list()
# 	if out_shape is None:
# 		if len(shape)==4:
# 			out_shape = shape[1:3]
# 		else:
# 			out_shape = shape[:2]
# 	return tf.contrib.image.transform(x, H, interpolation=interpolation, output_shape=out_shape)
#  
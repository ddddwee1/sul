import layers3 as L 
import tensorflow as tf 
import numpy as np 
import os 
from tensorflow.keras import Model as KModel

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
	if not one_hot:
		correct = tf.equal(tf.cast(tf.argmax(pred,-1),tf.int64),tf.cast(tf.argmax(y,-1),tf.int64))
	else:
		correct = tf.equal(tf.cast(tf.argmax(pred,-1),tf.int64),tf.cast(y,tf.int64))
	acc = tf.reduce_mean(tf.cast(correct,tf.float32))
	return acc

def weight_decay(wd, model):
	w_reg = wd * 0.5 * sum([tf.reduce_sum(tf.square(w)) for w in model.trainable_variables]) 
	return w_reg

################
# dumb model declaration
# make alias for init and call
class Model(KModel):
	def __init__(self, *args, **kwargs):
		super(Model, self).__init__()
		self.graph_initialized = False
		self.initialize(*args, **kwargs)

	def initialize(self, *args, **kwargs):
		pass 

	def call(self, x, *args, **kwargs):
		self.graph_initialized = True
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
	def __init__(self, size, outchn, dilation_rate=1, stride=1, pad='SAME', activation=-1, batch_norm=False, usebias=True, values=None):
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
		x = self.conv(x)
		if self.batch_norm:
			x = self.bn(x)
		if self.activation!=-1:
			x = self.act(x)
		return x 

class ConvLayer3D(KModel):
	def __init__(self, size, outchn, dilation_rate=1, stride=1, pad='SAME', activation=-1, batch_norm=False, usebias=True, values=None):
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

class OctConv(M.Model):
	def initialize(self, size, chn, ratio, input_ratio=None, stride=1, pad='SAME', activation=-1, batch_norm=False, usebias=True):
		chn_big = int(chn*ratio)
		chn_small = chn - chn_big
		self.avgpool = M.AvgPool(2)
		self.upsample = M.BilinearUpSample(2)
		self.Convhh = M.ConvLayer(size, chn_big, stride=stride, pad=pad, usebias=usebias)
		self.Convhl = M.ConvLayer(size, chn_small, stride=stride, pad=pad, usebias=usebias)
		self.Convlh = M.ConvLayer(size, chn_big, stride=stride, pad=pad, usebias=usebias)
		self.Convll = M.ConvLayer(size, chn_small, stride=stride, pad=pad, usebias=usebias)

		# bn and act
		if batch_norm:
			self.bn = L.batch_norm(values=values[idx:])
		if activation!=-1:
			self.act = L.activation(activation)

		self.batch_norm = batch_norm
		self.activation = activation
		self.chn_inp_big = chn_inp_big
		self.chn_inp_small = chn_inp_small
		self.ratio = ratio
		self.input_ratio = ratio if input_ratio is None else input_ratio
	
	def build(self, input_shape):
		chn = input_shape[-1]
		self.chn_inp_big = chn * self.input_ratio / (1 + self.input_ratio*3)
		self.chn_inp_small = chn - self.chn_inp_big

	def forward(self, x):
		big = x[:self.chn_inp_big*4]
		small = x[self.chn_inp_big*4:]
		big = tf.nn.depth_to_space(big, 2)

		hh = self.Convhh(big)
		ll = self.Convll(small)

		hl = self.avgpool(big)
		hl = self.Convhl(hl)
		
		lh = self.Convlh(small)
		lh = self.upsample(lh)

		h_out = hh + lh
		h_out = tf.nn.space_to_depth(h_out, 2) 
		l_out = ll + hl
		out = tf.concat([h_out, l_out], axis=-1)
		if self.batch_norm:
			out = self.bn(out)
		if self.activation!=-1:
			out = self.act(out)
		return out 

class OctMerge(M.Model):
	def initialize(self):
		self.avgpool = M.AvgPool(2)
	def forward(self,x):
		h = x 
		l = self.avgpool(x)
		h = tf.nn.space_to_depth(h, 2)
		out = tf.concat([h,l], axis=-1)
		return out 


###############
# alias for layers
AvgPool = L.avgpoolLayer
MaxPool = L.maxpoolLayer
GlobalAvgPool = L.globalAvgpoolLayer
flatten = L.flatten()
BatchNorm = L.batch_norm
DeconvLayer2D = DeconvLayer
ConvLayer2D = ConvLayer
BilinearUpSample = L.bilinearUpSample

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

######### Data Reader Template (parallel) ##########
# multi-process to read data
class DataReader():
	def __init__(self, data, fn, batch_size, shuffle=False, random_sample=False, processes=1, post_fn=None):
		from multiprocessing import Pool
		self.pool = Pool(processes)
		print('Starting parallel data loader...')
		self.process_fn = fn
		self.data = data
		self.batch_size = batch_size
		self.position = batch_size
		self.post_fn = post_fn
		self.random_sample = random_sample
		self.shuffle = shuffle
		if shuffle:
			random.shuffle(self.data)
		self._start_p(self.data[:batch_size])

	def _start_p(self, data):
		self.ps = []
		for i in data:
			self.ps.append(self.pool.apply_async(self.process_fn, [i]))

	def get_next(self):
		# print('call')
		# fetch data
		res = [i.get() for i in self.ps]

		# start new pre-fetch
		if self.random_sample:
			batch = random.sample(self.data, self.batch_size)
		else:
			if self.position + self.batch_size > len(self.data):
				self.position = 0
				if self.shuffle:
					random.shuffle(self.data)	
			batch = self.data[self.position:self.position+self.batch_size]
			self.position += self.batch_size
		
		self._start_p(batch)

		# post_process the data
		if self.post_fn is not None:
			res = self.post_fn(res)
		return res 

	def __next__(self):
		return self.get_next()

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
		self.grads = None
		self.grad_loss_fn = grad_loss_fn

	def compute_grad_loss(self, data, *args, **kwargs):
		# threads = []
		# pool = ThreadPool(processes=len(self.devices))

		# processes = []
		# pool = ThreadPool(processes=1)
		# for i in range(4):
		# 	with tf.device('/gpu:%d'%i):
		# 		pp = pool.apply_async(get_grads, (aa[i], model))
		# 		processes.append(pp)

		# # time.sleep(0.5)
		# rr = [p.get() for p in processes]

		rr = []
		
		for idx,i in enumerate(self.devices):
			with tf.device('/gpu:%d'%i):
				rr.append(self.grad_loss_fn(data[idx], *args, **kwargs))
		losses = []
		grads = [i[0] for i in rr]
		grads = [sum(g)/len(g) for g in zip(*grads)]
		for i in rr:
			losses.append(i[1])
		self.grads = grads
		return grads, losses

	def apply_grad(self, grads=None):
		if grads is None:
			grads = self.grads
		self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

	@tf.function
	def train_step(self, data):
		rr = []
		for idx,i in enumerate(self.devices):
			with tf.device('/gpu:%d'%i):
				rr.append(self.grad_loss_fn(data[idx], self.model))
				print('GPU:%d'%i)
		losses = []
		grads = [i[0] for i in rr]
		grads = [sum(g) for g in zip(*grads)]
		for i in rr:
			losses.append(i[1])
		self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
		return losses

	def split_data(self, data):
		res = []
		length = len(data[0])
		len_split = length//len(self.devices)
		for i in range(len(self.devices)):
			buff = []
			for j in range(len(data)):
				buff.append(data[j][i*len_split: min(length, i*len_split+len_split)])
			res.append(buff)
		return res 

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

def image_transform(x, H, out_shape=None, interpolation='BILINEAR'):
	# Will produce error if not specify 'output_shape' in eager mode
	shape = x.get_shape().as_list()
	if out_shape is None:
		if len(shape)==4:
			out_shape = shape[1:3]
		else:
			out_shape = shape[:2]
	return tf.contrib.image.transform(x, H, interpolation=interpolation, output_shape=out_shape)
 
def zip_grad(grads, vars):
	assert len(grads)==len(vars)
	grads_1 = []
	vars_1 = []
	for i in range(len(grads)):
		if not grads[i] is None:
			grads_1.append(grads[i])
			vars_1.append(vars[i])
	assert len(grads_1)!=0
	return zip(grads_1, vars_1)
import tensorflow as tf 
import numpy as np 

l_num = 0

var_dict = {}
var_list = []
###########################################################
#define weight and bias initialization

def weight(shape,record=True,dtype=None):
	context = tf.get_variable_scope().name
	name = context+'/weight'
	if name in var_dict:
		w = var_dict[name]
	else:
		w = tf.get_variable('weight',shape,initializer=tf.contrib.layers.xavier_initializer(),dtype=dtype)
		var_dict[name] = w 
		var_list.append(w)
		tf.add_to_collection('decay_variables',w)
	return w

def weight_conv(shape,dtype=None):
	context = tf.get_variable_scope().name
	name = context+'./kernel'
	if name in var_dict:
		k = var_dict[name]
	else:
		k = tf.get_variable('kernel',shape,initializer=tf.contrib.layers.xavier_initializer_conv2d(),dtype=dtype)
		var_dict[name] = k
		var_list.append(k)
		tf.add_to_collection('decay_variables',k)
	return k

def bias(shape,value=0.0,record=True,dtype=None):
	context = tf.get_variable_scope().name
	name = context+'/bias'
	if name in var_dict:
		b = var_dict[name]
	else:
		b = tf.get_variable('bias',shape,initializer=tf.constant_initializer(value),dtype=dtype)
		var_dict[name] = b
		var_list.append(b)
	return b

###########################################################
#define basic layers

def conv2D(x,size,outchn,name=None,stride=1,pad='SAME',usebias=True,kernel_data=None,bias_data=None,dilation_rate=1, weight_norm=False):
	global l_num
	print('Conv_bias:',usebias)
	if name is None:
		name = 'conv_l_'+str(l_num)
		l_num+=1
	# with tf.variable_scope(name):
	if isinstance(size,list):
		kernel = size
	else:
		kernel = [size,size]
	if (not kernel_data is None) and (not bias_data is None):
		z = tf.layers.conv2d(x, outchn, kernel, strides=(stride, stride), padding=pad,\
			dilation_rate=dilation_rate,\
			kernel_initializer=tf.constant_initializer(kernel_data),\
			use_bias=usebias,\
			bias_initializer=tf.constant_initializer(bias_data),name=name)
	else:
		z = tf.layers.conv2d(x, outchn, kernel, strides=(stride, stride), padding=pad,\
			dilation_rate=dilation_rate,\
			kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),\
			use_bias=usebias,\
			bias_initializer=tf.constant_initializer(0.0),name=name)
	return z

def conv3D(x,size,outchn,name=None,stride=1,pad='SAME',usebias=True,kernel_data=None,bias_data=None,dilation_rate=1):
	global l_num
	print('Conv_bias:',usebias)
	if name is None:
		name = 'conv_l_'+str(l_num)
		l_num+=1
	# with tf.variable_scope(name):
	if isinstance(stride,int):
		stride = [stride, stride, stride]
	if isinstance(size,list):
		kernel = size
	else:
		kernel = [size,size,size]
	if (not kernel_data is None) and (not bias_data is None):
		z = tf.layers.conv3d(x, outchn, kernel, strides=(stride[0], stride[1],stride[2]), padding=pad,\
			dilation_rate=dilation_rate,\
			kernel_initializer=tf.constant_initializer(kernel_data),\
			use_bias=usebias,\
			bias_initializer=tf.constant_initializer(bias_data),name=name)
	else:
		z = tf.layers.conv3d(x, outchn, kernel, strides=(stride[0], stride[1],stride[2]), padding=pad,\
			dilation_rate=dilation_rate,\
			kernel_initializer=tf.contrib.layers.xavier_initializer(),\
			use_bias=usebias,\
			bias_initializer=tf.constant_initializer(0.0),name=name)
	return z

# def conv2D(x,size,outchn,name=None,stride=1,pad='SAME',usebias=True,kernel_data=None,bias_data=None,dilation_rate=1,weight_norm=False):
# 	global l_num
# 	# print('Conv Bias:',usebias,'Weight norm:',weight_norm)
# 	inchannel = x.get_shape().as_list()[-1]
# 	# set name
# 	if name is None:
# 		name = 'conv_l_'+str(l_num)
# 		l_num+=1
# 	# set size
# 	if isinstance(size,list):
# 		size = [size[0],size[1],inchannel,outchn]
# 	else:
# 		size = [size, size, inchannel, outchn]
# 	# set stride
# 	if isinstance(stride,list):
# 		stride = [1,stride[0],stride[1],1]
# 	else:
# 		stride = [1,stride, stride, 1]
# 	# set dilation
# 	if isinstance(dilation_rate,list):
# 		dilation_rate = [1,dilation_rate[0],dilation_rate[1],1]
# 	else:
# 		dilation_rate = [1,dilation_rate,dilation_rate,1]

# 	with tf.variable_scope(name):
# 		if kernel_data:
# 			w = tf.constant(kernel_data,name='kernel')
# 		else:
# 			w = weight_conv(size)

		
# 		if weight_norm:
# 			print('Enable weight norm')
# 			w = w.initialized_value()
# 			w = tf.nn.l2_normalize(w, [0,1,2])
# 			try:
# 				s = tf.get_variable('weight_scale')
# 			except:
# 				print('Initialize weight norm')
# 				x_init = tf.nn.conv2d(x,w,stride,pad,dilations=dilation_rate)
# 				m_init, v_init = tf.nn.moments(x_init,[0,1,2])
# 				s_init = 1. / tf.sqrt(v_init + 1e-8)
# 				s = tf.get_variable('weight_scale',dtype=tf.float32,initializer=s_init)
# 				s = s.initialized_value()
# 			w = tf.reshape(s,[1,1,1,outchn]) *w
		
# 		out = tf.nn.conv2d(x,w,stride,pad,dilations=dilation_rate)

# 		if usebias:
# 			if bias_data:
# 				b = tf.constant(bias_data,name='bias')
# 			else:
# 				b = bias([outchn])
# 			out = tf.nn.bias_add(out,b)
# 	return out 


def sum(x,y):
	return x+y

def deconv2D(x,size,outchn,name,stride=1,pad='SAME'):
	with tf.variable_scope(name):
		if isinstance(size,list):
			kernel = size
		else:
			kernel = [size,size]
		z = tf.layers.conv2d_transpose(x, outchn, [size, size], strides=(stride, stride), padding=pad,\
			kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),\
			bias_initializer=tf.constant_initializer(0.1))
		return z

def conv2Ddw(x,inshape,size,multi,name,stride=1,pad='SAME',kernel_data=None,dtype=None,usebias=True):
	if dtype is None:
		dtype = x.dtype
	with tf.variable_scope(name):
		if isinstance(size,list):
			kernel = [size[0],size[1],inshape,multi]
		else:
			kernel = [size,size,inshape,multi]
		if kernel_data==None:
			w = weight(kernel,dtype=dtype)
		else:
			w = kernel_data
		res = tf.nn.depthwise_conv2d(x,w,[1,stride,stride,1],padding=pad)
		
		if usebias:
			b = bias([1,1,1,inshape*multi])
			res += b
	return res

def maxpooling(x,size,stride=None,name=None,pad='SAME'):
	global l_num
	if name is None:
		name = 'maxpooling_l_'+str(l_num)
		l_num+=1
	with tf.variable_scope(name):
		if stride is None:
			stride = size
		return tf.nn.max_pool(x,ksize=[1,size,size,1],strides=[1,stride,stride,1],padding=pad)

def avgpooling(x,size,stride=None,name=None,pad='SAME'):
	global l_num
	if name is None:
		name = 'avgpooling_l_'+str(l_num)
		l_num+=1
	with tf.variable_scope(name):
		if stride is None:
			stride = size
		return tf.nn.avg_pool(x,ksize=[1,size,size,1],strides=[1,stride,stride,1],padding=pad)

def maxpooling3d(x,size,stride=None,name=None,pad='SAME'):
	global l_num
	if name is None:
		name = 'maxpooling3d_l_'+str(l_num)
		l_num+=1
	with tf.variable_scope(name):
		if isinstance(stride,int):
			stride = [stride, stride, stride]
		if isinstance(size,int):
			size = [size, size, size]
		if stride is None:
			stride = size
		return tf.nn.max_pool3d(x,ksize=[1,size[0],size[1],size[2],1],strides=[1,stride[0],stride[1],stride[2],1],padding=pad)

def avgpooling3d(x,size,stride=None,name=None,pad='SAME'):
	global l_num
	if name is None:
		name = 'avgpooling3d_l_'+str(l_num)
		l_num+=1
	with tf.variable_scope(name):
		if isinstance(stride,int):
			stride = [stride, stride, stride]
		if isinstance(size,int):
			size = [size, size, size]
		if stride is None:
			stride = size
		return tf.nn.avg_pool3d(x,ksize=[1,size[0],size[1],size[2],1],strides=[1,stride[0],stride[1],stride[2],1],padding=pad)

def Fcnn(x,insize,outsize,name,activation=None,nobias=False,dtype=None):
	if dtype is None:
		dtype = tf.float32
	with tf.variable_scope(name):
		if nobias:
			print('No biased fully connected layer is used!')
			W = weight([insize,outsize],dtype=dtype)
			if activation==None:
				return tf.matmul(x,W)
			return activation(tf.matmul(x,W))
		else:
			W = weight([insize,outsize],dtype=dtype)
			b = bias([outsize],dtype=dtype)
			if activation==None:
				return tf.matmul(x,W)+b
			return activation(tf.matmul(x,W)+b)

def NALU(x,insize,outsize,name,activation=None):
	with tf.variable_scope(name):
		W = weight([insize, outsize])
		M = weight([insize, outsize]) 
		G = weight([insize, outsize]) # gate

		W = tf.tanh(W) * tf.sigmoid(M)

		g = tf.sigmoid(tf.matmul(x, G))
		m = tf.exp(tf.matmul(tf.log(tf.abs(x) + 1e-8), W))
		a = tf.matmul(x, W)

		out = g * a + (1. - g) * m 
	return out 


def MFM(x,half,name):
	with tf.variable_scope(name):
		#shape is in format [batchsize, x, y, channel]
		# shape = tf.shape(x)
		shape = x.get_shape().as_list()
		res = tf.reshape(x,[-1,shape[1],shape[2],2,shape[-1]//2])
		res = tf.reduce_max(res,axis=[3])
		return res

def MFMfc(x,half,name):
	with tf.variable_scope(name):
		shape = x.get_shape().as_list()
		# print('fcshape:',shape)
		res = tf.reduce_max(tf.reshape(x,[-1,2,shape[-1]//2]),reduction_indices=[1])
	return res

def accuracy(pred,y,name):
	with tf.variable_scope(name):
		correct = tf.equal(tf.cast(tf.argmax(pred,1),tf.int64),tf.cast(y,tf.int64))
		acc = tf.reduce_mean(tf.cast(correct,tf.float32))
		#acc = tf.cast(correct,tf.float32)
		return acc

def batch_norm(inp,name,epsilon=None,variance=None,training=True):
	print('BN training:',training)
	if not epsilon is None:
		return tf.layers.batch_normalization(inp,training=training,name=name,epsilon=epsilon)
	return tf.layers.batch_normalization(inp,training=training,name=name)

def lrelu(x,name,leaky=0.2):
	return tf.maximum(x,x*leaky,name=name)

def relu(inp,name):
	return tf.nn.relu(inp,name=name)

def tanh(inp,name):
	return tf.tanh(inp,name=name)

def elu(inp,name):
	return tf.nn.elu(inp,name=name)

def sigmoid(inp,name):
	return tf.sigmoid(inp,name=name)

def resize_nn(inp,size,name):
	with tf.name_scope(name):
		if isinstance(size,list):
			return tf.image.resize_nearest_neighbor(inp,size=(int(size[0]),int(size[1])))
		elif isinstance(size,tf.Tensor):
			return tf.image.resize_nearest_neighbor(inp,size=size)
		else:
			return tf.image.resize_nearest_neighbor(inp,size=(int(size),int(size)))

def upSampling(inp,multiplier,name):
	b,h,w,c = inp.get_shape().as_list()
	if isinstance(multiplier,list):
		h2 = h*multiplier[0]
		w2 = w*multiplier[1]
	else:
		h2 = h*multiplier
		w2 = w*multiplier
	return resize_nn(inp,[h2,w2],name)

# def shake_layer(x,a,b):
# 	return b*x + tf.stop_gradient(a*x - b*x)

@tf.custom_gradient
def shake_layer(x,a,b):
	y = tf.scalar_mul(a,x)
	def grad_fn(grad):
		return [tf.scalar_mul(b,grad),None,None]
	return y,grad_fn

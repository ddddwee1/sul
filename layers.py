import tensorflow as tf 
import numpy as np 

###########################################################
#define weight and bias initialization

def weight(shape,inp,outp):
	#Xavier initialization. To control the std-div of all layers
	return tf.get_variable('weight',shape,initializer=tf.truncated_normal_initializer())

def bias(shape,value=0.1):
	return tf.get_variable('bias',shape,initializer=tf.constant_initializer(value))

###########################################################
#define basic layers

def conv2D(x,size,outchn,name,stride=1,pad='SAME',activation=None):
	with tf.variable_scope(name):
		z = tf.layers.conv2d(x, outchn, [size, size], strides=(stride, stride), padding=pad)
		return z

def deconv2D(x,size,outchn,name,stride=1,pad='SAME'):
	with tf.variable_scope(name):
		z = tf.layers.conv2d_transpose(x, outchn, [size, size], strides=(stride, stride), padding=pad)
		return z

def maxpooling(x,size,name):
	with tf.variable_scope(name):
		return tf.nn.max_pool(x,ksize=[1,size,size,1],strides=[1,size,size,1],padding='SAME')

def avgpooling(x,size,name):
	with tf.variable_scope(name):
		return tf.nn.avg_pool(x,ksize=[1,size,size,1],strides=[1,size,size,1],padding='SAME')

def Fcnn(x,insize,outsize,name,activation=None,nobias=False):
	with tf.variable_scope(name):
		if nobias:
			print('No biased fully connected layer is used!')
			W = weight([insize,outsize],insize,outsize)
			tf.summary.histogram(name+'/weight',W)
			if activation==None:
				return tf.matmul(x,W)
			return activation(tf.matmul(x,W))
		else:
			W = weight([insize,outsize],insize,outsize)
			b = bias([outsize])
			tf.summary.histogram(name+'/weight',W)
			tf.summary.histogram(name+'/bias',b)
			if activation==None:
				return tf.matmul(x,W)+b
			return activation(tf.matmul(x,W)+b)

def MFM(x,name):
	with tf.variable_scope(name):
		#shape is in format [batchsize, x, y, channel]
		shape = tf.shape(x)
		res = tf.reshape(x,[shape[0],shape[1],shape[2],2,-1])
		res = tf.reduce_max(res,axis=[3])
		return res

def MFMfc(x,name):
	with tf.variable_scope(name):
		shape = tf.shape(x)
		res = tf.reduce_max(tf.reshape(x,[shape[0],2,-1]),reduction_indices=[1])
	return res

#Network in network
def NIN(x,inchn,outchn1,ksize,outchn2,name,stride=1):
	with tf.variable_scope(name):
		conv1_1 = conv2D(x,1,inchn,outchn1,name=name+'_1x1')
		mfm1 = MFM(conv1_1,name=name+'_mfm1')
		conv2 = conv2D(mfm1,ksize,int(outchn1/2),outchn2,stride=stride,name=name+'_'+str(ksize)+'x'+str(ksize))
		return MFM(conv2,name=name+'_mfm2')

def accuracy(pred,y,name):
	with tf.variable_scope(name):
		correct = tf.equal(tf.cast(tf.argmax(pred,1),tf.int32),y)
		acc = tf.reduce_mean(tf.cast(correct,tf.float32))
		return acc

def batch_norm(inp,name):
	return tf.layers.batch_normalization(inp,training=True,name=name)

def lrelu(x,name,leaky=0.2):
	return tf.maximum(x,x*leaky,name=name)

def relu(inp,name):
	return tf.nn.relu(inp,name=name)

def tanh(inp,name):
	return tf.tanh(inp,name=name)

def elu(inp,name):
	return tf.nn.elu(inp,name=name)
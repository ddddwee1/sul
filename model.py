import layers as L 
import tensorflow as tf
import numpy as np 
import os 
import time 

acc = -1

PARAM_RELU = 0
PARAM_LRELU = 1
PARAM_ELU = 2
PARAM_TANH = 3
PARAM_MFM = 4
PARAM_MFM_FC = 5
PARAM_SIGMOID = 6

VAR_LIST = L.var_list

def set_gpu(config_str):
	import os
	os.environ["CUDA_VISIBLE_DEVICES"] = config_str

def loadSess(modelpath=None,sess=None,modpath=None,mods=None,var_list=None,init=False, init_dict=None):
#load session if there exist any models, and initialize the sess if not
	assert modpath==None or mods==None
	assert (not modelpath==None) or (not modpath==None) or (not modpath==None)
	if sess==None:
		sess = tf.Session()
	if init:
		if not os.path.exists(modelpath):
			os.mkdir(modelpath)
		print('Initializing...')
		sess.run(tf.global_variables_initializer(),feed_dict=init_dict)
	if var_list==None:
		saver = tf.train.Saver()
	else:
		saver = tf.train.Saver(var_list)
	
	if modpath!=None:
		mod = modpath
		print('loading from model:',mod)
		saver.restore(sess,mod)
	elif mods!=None:
		for m in mods:
			print('loading from model:',m)
			saver.restore(sess,m)
	elif modelpath!=None:
		ckpt = tf.train.get_checkpoint_state(modelpath)
		if ckpt:
			mod = ckpt.model_checkpoint_path
			print('loading from model:',mod)
			saver.restore(sess,mod)
		else:
			print('No checkpoint in folder, use initial graph...')
	return sess

def initialize(sess, init_dict=None):
	sess.run(tf.global_variables_initializer(), feed_dict=init_dict)

def accuracy(inp,lab):
	global acc
	acc +=1
	return L.accuracy(inp,lab,'accuracy_'+str(acc))

def enforcedClassifier(featurelayer,lbholder,dropout=1,multi=None,L2norm=False,L2const=10.0):
	with tf.variable_scope('Enforced_Softmax'):
		inp_shape = featurelayer.get_shape().as_list()
		inputdim = inp_shape[1]
		featurelayer = tf.nn.dropout(featurelayer,dropout)
		CLASS = lbholder.get_shape().as_list()[-1]
		w = L.weight([inputdim,CLASS])
		if L2norm:
			nfl = tf.nn.l2_normalize(featurelayer,1)
			buff = tf.matmul(nfl,tf.nn.l2_normalize(w,0))
			evallayer = tf.scalar_mul(L2const,buff)
		else:
			buff = tf.matmul(featurelayer,w)
			evallayer = tf.matmul(featurelayer,w)
		floatlb = tf.cast(lbholder,tf.float32)
		lbc = tf.ones_like(lbholder) - floatlb
		filteredmtx = tf.multiply(lbc,buff)
		#filteredmtx = tf.maximum(filteredmtx*1.2,filteredmtx*0.8)
		cosmtx = tf.multiply(floatlb,buff)
		if multi is not None:
			cosmtx = (tf.minimum(cosmtx*multi[0],cosmtx*multi[1]))*floatlb
		lstlayer = cosmtx+filteredmtx
		if L2norm:
			lstlayer = tf.scalar_mul(L2const, lstlayer)
	return lstlayer,evallayer

def get_feed_dict(keylist,vallist):
	assert len(keylist)==len(vallist)
	d = {}
	for i in range(len(keylist)):
		# print(keylist[i],'\t',type(vallist))
		d[keylist[i]] = vallist[i]
	return d

def runSess(sess,tensorlist,feeddict=None):
	return sess.run(tensorlist,feed_dict=feeddict)

def get_trainable_vars(scope=None):
	return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope)

def get_all_vars(scope=None):
	return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=scope)

def get_update_ops(scope=None):
	return tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope=scope)

def get_var_decay(rate,scope=None):
	with tf.variable_scope('weight_decay'):
		w = tf.get_collection('decay_variables',scope=scope)
		decay_ops = [tf.assign_sub( v , (1.-rate)*v) for v in w]
		with tf.control_dependencies(decay_ops):
			decay_op = tf.no_op()
	return decay_op

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

class Model():
	def __init__(self,inp,size=None):
		self.result = inp
		if size is None:
			self.inpsize = inp.get_shape().as_list()
		else:
			self.inpsize = list(size)
		self.layernum = 0
		self.bntraining = True
		self.epsilon = None

	def set_bn_training(self,training):
		self.bntraining = training

	def set_bn_epsilon(self,epsilon):
		self.epsilon = epsilon

	def get_current(self):
		return self.get_current_layer()

	def get_current_layer(self):
		return self.result

	def __call__(self):
		return [self.result,self.inpsize]

	def get_shape(self):
		return self.inpsize

	def activation(self,param):
		return self.activate(param)

	def activate(self,param):
		inp = self.result
		with tf.name_scope('activation_'+str(self.layernum)):
			if param == 0:
				res =  L.relu(inp,name='relu_'+str(self.layernum))
			elif param == 1:
				res =  L.lrelu(inp,name='lrelu_'+str(self.layernum))
			elif param == 2:
				res =  L.elu(inp,name='elu_'+str(self.layernum))
			elif param == 3:
				res =  L.tanh(inp,name='tanh_'+str(self.layernum))
			elif param == 4:
				self.inpsize[-1] = self.inpsize[-1]//2
				res =  L.MFM(inp,self.inpsize[-1],name='mfm_'+str(self.layernum))
			elif param == 5:
				self.inpsize[-1] = self.inpsize[-1]//2
				res =  L.MFMfc(inp,self.inpsize[-1],name='mfm_'+str(self.layernum))
			elif param == 6:
				res =  L.sigmoid(inp,name='sigmoid_'+str(self.layernum))
			else:
				res =  inp
		self.result = res
		return self.result

	def convLayer(self,size,outchn,dilation_rate=1,stride=1,pad='SAME',activation=-1,batch_norm=False,layerin=None,usebias=True,kernel_data=None,bias_data=None,weight_norm=False):
		with tf.variable_scope('conv_'+str(self.layernum)):
			if isinstance(size,list):
				kernel = size
			else:
				kernel = [size,size]
			if layerin!=None:
				self.result = layerin
				self.inpsize = layerin.get_shape().as_list()
			self.result = L.conv2D(self.result,kernel,outchn,'conv_'+str(self.layernum),stride=stride,pad=pad,usebias=usebias,kernel_data=kernel_data,bias_data=bias_data,dilation_rate=dilation_rate,weight_norm=weight_norm)
			if batch_norm:
				self.result = L.batch_norm(self.result,'batch_norm_'+str(self.layernum),training=self.bntraining,epsilon=self.epsilon)
			self.layernum += 1
			self.inpsize = self.result.get_shape().as_list()
			self.activate(activation)
		return self.result

	def dwconvLayer(self,kernel,multi,stride=1,pad='SAME',activation=-1,batch_norm=False,weight=None,usebias=True):
		with tf.variable_scope('dwconv_'+str(self.layernum)):
			if isinstance(kernel,list):
				kernel = kernel
			else:
				kernel = [kernel,kernel]
			self.result = L.conv2Ddw(self.result,self.inpsize[3],kernel,multi,'dwconv_'+str(self.layernum),stride=stride,pad=pad,weight_data=weight,usebias=usebias)
			if batch_norm:
				self.result = L.batch_norm(self.result,'batch_norm_'+str(self.layernum),training=self.bntraining,epsilon=self.epsilon)
			self.layernum+=1
			self.inpsize = self.result.get_shape().as_list()
			self.activate(activation)
		return self.result

	def spconvLayer(self,size,multi,stride=1,pad='SAME',activation=-1,batch_norm=False):
		self.dwconvLayer(size,multi,stride=stride,pad=pad)
		self.convLayer(1,self.inpsize[3],activation=activation,batch_norm=batch_norm)
		return self.result

	def deconvLayer(self,kernel,outchn,stride=1,pad='SAME',activation=-1,batch_norm=False):
		self.result = L.deconv2D(self.result,kernel,outchn,'deconv_'+str(self.layernum),stride=stride,pad=pad)
		if batch_norm:
			self.result = L.batch_norm(self.result,'batch_norm_'+str(self.layernum),training=self.bntraining,epsilon=self.epsilon)
		self.layernum+=1
		self.inpsize = self.result.get_shape().as_list()
		self.activate(activation)
		return self.result

	def maxpoolLayer(self,size,stride=None,pad='SAME'):
		if stride==None:
			stride = size
		self.result = L.maxpooling(self.result,size,stride,'maxpool_'+str(self.layernum),pad=pad)
		self.inpsize = self.result.get_shape().as_list()
		return self.result

	def avgpoolLayer(self,size,stride=None,pad='SAME'):
		if stride==None:
			stride = size
		self.result = L.avgpooling(self.result,size,stride,'maxpool_'+str(self.layernum),pad=pad)
		self.inpsize = self.result.get_shape().as_list()
		return self.result

	def flatten(self):
		self.result = tf.reshape(self.result,[-1,self.inpsize[1]*self.inpsize[2]*self.inpsize[3]])
		self.transShape = [self.inpsize[1],self.inpsize[2],self.inpsize[3],0]
		self.inpsize = [None,self.inpsize[1]*self.inpsize[2]*self.inpsize[3]]
		return self.result

	def construct(self,shape):
		self.result = tf.reshape(self.result,[-1,shape[0],shape[1],shape[2]])
		self.inpsize = [None,shape[0],shape[1],shape[2]]
		return self.result

	def fcLayer(self,outsize,activation=-1,nobias=False,batch_norm=False):
		with tf.variable_scope('fc_'+str(self.layernum)):
			self.inpsize = [i for i in self.inpsize]
			self.result = L.Fcnn(self.result,self.inpsize[1],outsize,'fc_'+str(self.layernum),nobias=nobias)
			if batch_norm:
				self.result = L.batch_norm(self.result,'batch_norm_'+str(self.layernum),training=self.bntraining,epsilon=self.epsilon)
			self.inpsize[1] = outsize
			self.activate(activation)
			self.layernum+=1
		return self.result

	def multiply(self,layerin):
		if isinstance(layerin,list):
			self.result = self.result*layerin[0]
		else:
			self.result = self.result*layerin

	def sum(self,layerin):
		with tf.variable_scope('sum_'+str(self.layernum)):
			self.result = self.result +	layerin
		return self.result

	def NIN(self,size,outchn1,outchn2,activation=-1,batch_norm=False,pad='SAME'):
		with tf.variable_scope('NIN_'+str(self.layernum)):
			self.convLayer(1,outchn1,activation=activation,batch_norm=batch_norm)
			self.convLayer(size,outchn2,activation=activation,batch_norm=batch_norm,pad=pad)
		return self.result

	def incep(self,outchn1,outchn2,outchn3,outchn4,outchn5,activation=-1,batch_norm=False):
		with tf.variable_scope('Incep_'+str(self.layernum)):
			orignres = self.result
			orignsize = self.inpsize
			a,_ = self.NIN(3,outchn1,outchn2,activation=activation,batch_norm=batch_norm)
			self.result = orignres
			b,_ = self.NIN(5,outchn3,outchn4,activation=activation,batch_norm=batch_norm)
			self.result = orignres
			c,_ = self.convLayer(1,outchn5,activation=activation,batch_norm=batch_norm)
			csize = self.inpsize
			self.result = tf.concat(axis=3,values=[a,b,c])
			self.inpsize = self.result.get_shape().as_list()
			return self.result

	def concat_to_current(self,layerin,axis=3):
		with tf.variable_scope('concat'+str(self.layernum)):
			self.result = tf.concat(axis=axis,values=[self.result,layerin])
			self.inpsize = self.result.get_shape().as_list()
		return self.result

	def concat_to_all_batch(self,layerinfo):
		with tf.variable_scope('concat'+str(self.layernum)):
			layerin = layerinfo
			layerin = tf.expand_dims(layerin,0)
			layerin = tf.tile(layerin,[tf.shape(self.result)[0],1,1,1])
			self.result = tf.concat(axis=-1,values=[self.result,layerin])
			self.inpsize = self.result.get_shape().as_list()
		return self.result

	def set_current(self,layerinfo):
		if isinstance(layerinfo,list):
			self.result = layerinfo[0]
			self.inpsize = layerinfo[1]
		else:
			self.result = layerinfo
			self.inpsize = self.result.get_shape().as_list()

	def set_current_layer(self,layerinfo):
		self.set_current(layerinfo)

	def dropout(self,ratio):
		with tf.name_scope('dropout'+str(self.layernum)):
			self.result = tf.nn.dropout(self.result,ratio)
		return self.result

	def l2norm(self,axis=1):
		with tf.name_scope('l2norm'+str(self.layernum)):
			self.result = tf.nn.l2_normalize(self.result,axis)
		return self.result

	def batch_norm(self):
		with tf.variable_scope('batch_norm'+str(self.layernum)):
			self.result = L.batch_norm(self.result,'batch_norm_'+str(self.layernum),training=self.bntraining,epsilon=self.epsilon)
		return self.result

	def resize_nn(self,multip):
		assert self.inpsize[1] == self.inpsize[2]
		with tf.variable_scope('resize_'+str(self.layernum)):
			self.result = L.resize_nn(self.result,multip*self.inpsize[1],name='resize_nn_'+str(self.layernum))
			self.inpsize[1] *= multip
			self.inpsize[2] *= multip
		return self.result

	def reshape(self,shape):
		with tf.variable_scope('reshape_'+str(self.layernum)):
			self.result = tf.reshape(self.result,shape)
			self.inpsize = shape
		return self.result

	def transpose(self,order):
		with tf.variable_scope('transpose_'+str(self.layernum)):
			self.result=tf.transpose(self.result,order)
			self.inpsize = [self.inpsize[i] for i in order]
		return self.result

	def gradient_flip_layer(self):
		with tf.variable_scope('Gradient_flip_'+str(self.layernum)):
			@tf.RegisterGradient("GradFlip")
			def _flip_grad(op,grad):
				return [tf.negative(grad)]

			g = tf.get_default_graph()
			with g.gradient_override_map({'Identity':'GradFlip'}):
				self.result = tf.identity(self.result)
		return self.result

	def pyrDown(self,stride=1):
		with tf.variable_scope('Pyramid_down_'+str(self.layernum)):
			kernel = np.float32([\
				[1,4 ,6 ,4 ,1],\
				[4,16,24,16,4],\
				[6,24,36,24,6],\
				[4,16,24,16,4],\
				[1,4 ,6 ,4 ,1]])/256.0
			channel = self.inpsize[3]
			kernel = np.repeat(kernel[:,:,np.newaxis],channel,axis=2)
			kernel = np.expand_dims(kernel,axis=3)
			kernel = tf.constant(kernel,dtype=tf.float32)
			with tf.name_scope('gaussian_conv'):
				self.result = tf.nn.depthwise_conv2d(self.result,kernel,[1,stride,stride,1],'SAME')
		return self.result

	def primaryCaps(self, size, vec_dim, n_chn,activation=None, stride=1,pad='SAME'):
		with tf.variable_scope('Caps_'+str(self.layernum)):
			self.convLayer(size, vec_dim*n_chn, activation=activation, stride=stride, pad=pad)
			shape = self.result.get_shape().as_list()
			# output: BSIZE, capin, 1, vdim, 1
			self.result = tf.reshape(self.result, [-1,shape[1]*shape[2]*shape[3]//vec_dim,1,vec_dim,1])
			self.inpsize = self.result.get_shape().as_list()
			self.squash()
		return self.result

	def squash(self):
		with tf.variable_scope('squash_'+str(self.layernum)):
			sqr = tf.reduce_sum(tf.square(self.result),-2,keep_dims=True)
			activate = sqr / (1+sqr)
			self.result = activate * tf.nn.l2_normalize(self.result,-2)
		return self.result

	def capsLayer(self,outchn,vdim2,iter_num,BSIZE=None):
		if BSIZE is None:
			BSIZE = self.result.get_shape().as_list()
		with tf.variable_scope('capLayer_'+str(self.layernum)):
			# input size: [BSIZE, capin, 1, vdim1,1]
			_,capin,_,vdim1,_ = self.inpsize
			W = L.weight([1,capin,outchn,vdim1,vdim2])
			W = tf.tile(W,[BSIZE,1,1,1,1])
			b = tf.constant(0,dtype=tf.float32,shape=[BSIZE,capin,outchn,1,1])
			res_tile = tf.tile(self.result,[1,1,outchn,1,1])
			res = tf.matmul(W,res_tile,transpose_a=True)  # [BSIZE, capin, capout, vdim2, 1]
			for i in range(iter_num):
				with tf.variable_scope('Routing_'+str(self.layernum)+'_'+str(i)):
					c = tf.nn.softmax(b,dim=2)
					self.result = tf.reduce_sum(c*res,1,keep_dims=True)  # [BSIZE, 1, capout, vdim2, 1]
					self.squash()
					if i!=iter_num-1:
						b = tf.reduce_sum(self.result * res, -2, keep_dims=True)
			self.result = tf.einsum('ijklm->ikjlm',self.result)
			self.inpsize = [None,outchn,1,vdim2,1]
			self.layernum += 1
		return self.result

	def capsDown(self):
		with tf.variable_scope('Caps_Dim_Down_'+str(self.layernum)):
			self.result = tf.reduce_sum(self.result,-1)
			self.result = tf.reduce_sum(self.result,-2)
			self.inpsize = [None,self.inpsize[1],self.inpsize[3]]
		return self.result

	def capsMask(self,labholder):
		with tf.variable_scope('capsMask_'+str(self.layernum)):
			labholder = tf.expand_dims(labholder,-1)
			self.result = self.result * labholder
			self.result = tf.reshape(self.result,[-1,self.inpsize[1]*self.inpsize[2]])
			self.inpsize = [None,self.inpsize[1]*self.inpsize[2]]
		return self.result

	def pad(self,padding):
		with tf.variable_scope('pad_'+str(self.layernum)):
			# left, right, top, btm
			if isinstance(padding,list):
				self.result = tf.pad(self.result,[[0,0],[padding[0],padding[1]],[padding[2],padding[3]],[0,0]])
			else:
				self.result = tf.pad(self.result,[[0,0],[padding,padding],[padding,padding],[0,0]])
			self.inpsize = self.result.get_shape().as_list()
		return self.result

	def caps_conv(self,ksize,outdim,outcaps,stride=1,activation='l2',usebias=True):
		print('Caps_conv_bias:',usebias)
		# resize the input to [BSIZE, height, width, capsnum, vecdim]
		capsnum = self.inpsize[3]
		vecdim = self.inpsize[4]
		stride_ = [1,stride,stride,capsnum,1]
		with tf.variable_scope('CapsConv_'+str(self.layernum)):
			res = []
			for i in range(outcaps):
				with tf.variable_scope('CapsConv_3dConv_'+str(i)):
					k = L.weight([ksize,ksize,capsnum,vecdim,outdim])
					buff = tf.nn.conv3d(self.result , k , stride_ , 'SAME')
					res.append(buff)
			self.result = tf.concat(res, axis=3)
			if usebias:
				b = L.bias([1,1,1,outcaps,outdim])
				self.result += b
			if activation=='l2':
				self.result = tf.nn.l2_normalize(self.result,-1)
		self.layernum += 1
		self.inpsize = self.result.get_shape().as_list()
		
		return self.result

	def capsulization(self,dim,caps):
		bsize,h,w,chn = self.inpsize
		assert dim*caps==chn,'Dimension and capsule number must be complemented with channel number'
		self.result = tf.reshape(self.result,[-1,h,w,caps,dim])
		self.inpsize = self.result.get_shape().as_list()
		return self.result

	def caps_flatten(self):
		bsize,h,w,caps,dim = self.inpsize
		self.result = tf.reshape(self.result,[-1,h*w*caps,1,dim,1])
		self.inpsize = self.result.get_shape().as_list()
		return self.result

	def SelfAttention(self,att_num=None,is_fc=False,residual=False):
		assert is_fc or att_num, 'must state attention feature num for conv'
		def flatten_hw(layer):
			shape = layer.get_shape().as_list()
			layer = tf.reshape(layer,[-1,shape[1]*shape[2],shape[3]])
			return layer

		with tf.variable_scope('att_'+str(self.layernum)):
			# conv each of them
			current = self.result
			current_shape = current.get_shape().as_list()
			orig_num = current_shape[-1]
			if is_fc:
				f = L.Fcnn(current,orig_num,'att_fc_f'+str(self.layernum))
				g = L.Fcnn(current,orig_num,'att_fc_g'+str(self.layernum))
				h = L.Fcnn(current,orig_num,'att_fc_h'+str(self.layernum))
				f = tf.expand_dims(f,axis=-1)
				g = tf.expand_dims(g,axis=-1)
				h = tf.expand_dims(h,axis=-1)
			else:
				f = L.conv2D(current,1,att_num,'att_conv_f_'+str(self.layernum))
				g = L.conv2D(current,1,att_num,'att_conv_g_'+str(self.layernum))
				h = L.conv2D(current,1,orig_num,'att_conv_h_'+str(self.layernum))

				# flatten them
				f = flatten_hw(f)
				g = flatten_hw(g)
				h = flatten_hw(h)

			# softmax(fg)
			fg = tf.matmul(f,g,transpose_b=True)
			fg = tf.nn.softmax(fg,-1)

			# out = scale(softmax(fg)h) + x 
			scale = tf.get_variable('Variable',shape=[],initializer=tf.constant_initializer(0.0))
			out = tf.matmul(fg,h)
			if is_fc:
				out = tf.reshape(out,[-1,orig_num])
			else:
				out = tf.reshape(out,[-1]+current_shape[1:3]+[orig_num])
			if residual:
				out = out + current
			self.layernum+=1
			self.inpsize = out.get_shape().as_list()
			self.result = out
		return self.result

	def res_block(self,output,stride=1,ratio=4,activation=PARAM_RELU,batch_norm=True, input_batch_norm=True):
		with tf.variable_scope('block'+str(self.layernum)):
			inp = self.result.get_shape().as_list()[-1]
			aa = self.result
			if inp==output:
				if stride==1:
					l0 = self.get_current()
				else:
					l0 = self.maxpoolLayer(stride)
			else:
				l0 = self.convLayer(1,output,activation=activation,stride=stride)
			self.set_current_layer(aa)
			if batch_norm and input_batch_norm:
				self.batch_norm()
			self.activate(activation)
			self.convLayer(1,output//ratio,activation=activation,batch_norm=batch_norm)
			self.convLayer(3,output//ratio,activation=activation,batch_norm=batch_norm,stride=stride)
			self.convLayer(1,output)
			self.sum(l0)
		return self.result

	def shake_layer(self,a,b):
		with tf.variable_scope('shake_layer'+str(self.layernum)):
			self.result = L.shake_layer(self.result,a,b)
			self.layernum += 1
		return self.result 

	def shake_block(self,output,stride=1,ratio=4,activation=PARAM_RELU,batch_norm=True, group=2, is_training=True):
		with tf.variable_scope('shake_block'+str(self.layernum)):
			print('Shake block training:',is_training)
			if is_training:
				a = tf.random_uniform([],minval=0.,maxval=1.)
				b = tf.random_uniform([],minval=0.,maxval=1.)
			else:
				a = b = 1/float(group)
			inp = self.result.get_shape().as_list()[-1]
			aa = self.result
			if inp==output:
				if stride==1:
					l0 = self.get_current()
				else:
					l0 = self.maxpoolLayer(stride)
			else:
				l0 = self.convLayer(1,output,activation=activation,stride=stride)
			self.set_current_layer(aa)
			if batch_norm:
				self.batch_norm()
			bb = self.activate(activation)
			self.convLayer(1,output//ratio,activation=activation,batch_norm=batch_norm)
			self.convLayer(3,output//ratio,activation=activation,batch_norm=batch_norm,stride=stride)
			self.convLayer(1,output)
			branch1 = self.shake_layer(a, b)

			self.set_current_layer(bb)
			self.convLayer(1,output//ratio,activation=activation,batch_norm=batch_norm)
			self.convLayer(3,output//ratio,activation=activation,batch_norm=batch_norm,stride=stride)
			self.convLayer(1,output)
			branch2 = self.shake_layer(1.-a, 1.-b)

			self.sum(branch1)
			self.sum(l0)
		return self.result

	def QAttention(self,feature):
		with tf.variable_scope('Q_attention_'+str(self.layernum)):
			self.result = tf.expand_dims(self.result,-1) 
			e = tf.matmul(feature, self.result) # [bsize, feature_num, 1]
			e = tf.squeeze(e,[-1])
			e = tf.nn.softmax(e,-1)
			out = e * self.result
			out = tf.reduce_mean(out,1)
			self.result = out 
			self.inpsize = self.result.get_shape().as_list()
		return self.result

	

# -------------- LSTM related functions & classes ----------------
# Provide 3 types of LSTM for different usage.
def LSTM(inp_holder, hidden_holder, state_holder,outdim,name,reuse=False):
	with tf.variable_scope(name,reuse=reuse):
		inp = tf.concat([inp_holder,hidden_holder],-1)
		inpdim = inp.get_shape().as_list()[-1]
		
		# info 
		I1 = L.Fcnn(inp,inpdim, outdim, name='Info_1')
		I2 = L.Fcnn(inp,inpdim, outdim, name='Info_2')
		# forget
		F = L.Fcnn(inp,inpdim, outdim, name='Forget')
		# output
		O = L.Fcnn(inp,inpdim, outdim, name='Output')

		I1_h = L.Fcnn(hidden_holder, outdim, outdim, name='Info_1_hid')
		I2_h = L.Fcnn(hidden_holder, outdim, outdim, name='Info_2_hid')

		F_h = L.Fcnn(hidden_holder, outdim, outdim, name='Forget_hid')
		O_h = L.Fcnn(hidden_holder, outdim, outdim, name='Output_hid')

		I = tf.sigmoid(I1 + I1_h) * tf.tanh(I2 + I2_h)
		F = tf.sigmoid(F + F_h)

		C_next = F * state_holder + I
		O = tf.sigmoid(O + O_h)

		H = O * tf.tanh(C_next)

	return H,C_next

class BasicLSTM():
	def __init__(self,dim,name):
		self.reuse = False
		self.name = name
		self.dim = dim

	def apply(self, inp, hidden=None, cell=None):
		if hidden is None or cell is None:
			with tf.variable_scope('Initial_zero_state'):
				bsize = tf.shape(inp)[0]
				zero_state = tf.constant(0., shape=[1,self.dim])
				zero_state = tf.tile(zero_state, [bsize, 1])
		if hidden is None:
			hidden = zero_state
		if cell is None:
			cell = zero_state
		out = LSTM(inp, hidden, cell, self.dim, self.name, self.reuse)
		self.reuse = True
		return out

LSTM_num = 0
class SimpleLSTM():
	def __init__(self,dim, out_func=None, init_hidden=None, init_cell=None):
		global LSTM_num
		self.name = 'LSTM_%d'%LSTM_num
		self.lstm = BasicLSTM(dim, self.name)
		LSTM_num += 1
		self.out_reuse = False
		self.out_func = out_func
		self.hidden = init_hidden
		self.cell = init_cell

	def apply(self, inp):
		with tf.variable_scope(self.name):
			inp_split = tf.unstack(inp, axis=1)
			lstm_outputs = []
			for i in range(len(inp_split)):
				self.hidden, self.cell = self.lstm.apply(inp_split[i], self.hidden, self.cell)
				lstm_outputs.append(self.hidden)
			if self.out_func is None:
				outputs = lstm_outputs
			else:
				outputs = []
				for out in lstm_outputs:
					o = self.out_func(out, reuse=self.out_reuse)
					self.out_reuse = True
					outputs.append(o)
			out = tf.stack(outputs,1)
		return out

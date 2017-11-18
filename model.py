import layers as L 
import tensorflow as tf
import copy
import numpy as np 

acc = -1

PARAM_RELU = 0
PARAM_LRELU = 1
PARAM_ELU = 2
PARAM_TANH = 3
PARAM_MFM = 4
PARAM_MFM_FC = 5
PARAM_SIGMOID = 6

def loadSess(modelpath=None,sess=None,modpath=None,mods=None,var_list=None,init=False):
#load session if there exist any models, and initialize the sess if not
	assert modpath==None or mods==None
	assert (not modelpath==None) or (not modpath==None) or (not modpath==None)
	if sess==None:
		sess = tf.Session()
	if init:
		sess.run(tf.global_variables_initializer())
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
			sess.run(tf.global_variables_initializer())
			print('No checkpoint in folder, use initial graph...')
	return sess

def accuracy(inp,lab):
	global acc
	acc +=1
	return L.accuracy(inp,lab,'accuracy_'+str(acc))

def enforcedClassifier(featurelayer,CLASS,BSIZE,lbholder,dropout=1,enforced=False,L2norm=False,L2const=10.0):
	with tf.variable_scope('Enforced_Softmax1'):
		if enforced:
			print('Enforced softmax loss is enabled.')
	with tf.variable_scope('Enforced_Softmax'):
		inp_shape = featurelayer.get_shape().as_list()
		inputdim = inp_shape[1]
		featurelayer = tf.nn.dropout(featurelayer,dropout)
		w = L.weight([inputdim,CLASS])
		nfl = tf.nn.l2_normalize(featurelayer,1)
		buff = tf.matmul(nfl,tf.nn.l2_normalize(w,0))
		if L2norm:
			evallayer = tf.scalar_mul(L2const,buff)
		else:
			evallayer = tf.matmul(featurelayer,w)
		if enforced:
			floatlb = tf.cast(lbholder,tf.float32)
			lbc = tf.ones([BSIZE,CLASS],dtype=tf.float32) - floatlb
			filteredmtx = tf.multiply(lbc,buff)
			#filteredmtx = tf.maximum(filteredmtx*1.2,filteredmtx*0.8)
			cosmtx = tf.multiply(floatlb,buff)
			cosmtx2 = (tf.minimum(cosmtx*0.9,cosmtx*1.1))*floatlb
			lstlayer = cosmtx2+filteredmtx
			if not L2norm:
				nb = tf.norm(w,axis=0,keep_dims=True)
				nf = tf.norm(featurelayer,axis=1,keep_dims=True)
				lstlayer = nb*lstlayer
				lstlayer = nf*lstlayer
		else:
			lstlayer = evallayer
	return lstlayer,evallayer

# def enforcedClassfier2(featurelayer,inputdim,lbholder,BSIZE,CLASS,enforced=False,dropout=1):
# 	with tf.variable_scope('Enforced_Softmax2'):
# 		if enforced:
# 			print('Enforced softmax loss is enabled.')
# 		featurelayer = tf.nn.dropout(featurelayer,dropout)
# 		w = L.weight([inputdim,CLASS])
# 		nfl = tf.nn.l2_normalize(featurelayer,1)
# 		buff = tf.matmul(nfl,tf.nn.l2_normalize(w,0))
# 		constant = 40.0
# 		evallayer = tf.scalar_mul(constant,buff)
# 		if enforced:
# 			floatlb = tf.cast(lbholder,tf.float32)
# 			lbc = tf.ones([BSIZE,CLASS],dtype=tf.float32) - floatlb
# 			filteredmtx = tf.multiply(lbc,evallayer)
# 			#filteredmtx = tf.maximum(filteredmtx*1.2,filteredmtx*0.8)
# 			cosmtx = tf.multiply(floatlb,evallayer)
# 			cosmtx2 = (tf.minimum(cosmtx*0.8,cosmtx*1.2))*floatlb
# 			lstlayer = cosmtx2+filteredmtx
# 		else:
# 			lstlayer = evallayer
# 	return lstlayer,evallayer

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

def get_update_ops(scope=None):
	return tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope=scope)

class Model():
	def __init__(self,inp,size):
		self.result = inp
		self.inpsize = list(size)
		self.layernum = 0
		self.transShape = None
		self.varlist = []
		self.fcs = []
		self.bntraining = True

	def set_bn_training(self,training):
		self.bntraining = training

	def get_current_layer(self):
		return self.result

	def get_shape(self):
		return self.inpsize

	def get_current(self):
		return [self.result,list(self.inpsize)]

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
		return [self.result,list(self.inpsize)]

	def convLayer(self,size,outchn,stride=1,pad='SAME',activation=-1,batch_norm=False,layerin=None,usebias=True,kernel_data=None,bias_data=None):
		with tf.variable_scope('conv_'+str(self.layernum)):
			if isinstance(size,list):
				kernel = size
			else:
				kernel = [size,size]
			if layerin!=None:
				self.result=layerin[0]
				self.inpsize=list(layerin[1])
			self.result = L.conv2D(self.result,kernel,outchn,'conv_'+str(self.layernum),stride=stride,pad=pad,usebias=usebias,kernel_data=kernel_data,bias_data=bias_data)
			self.varlist = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
			if batch_norm:
				self.result = L.batch_norm(self.result,'batch_norm_'+str(self.layernum),training=self.bntraining)
			self.layernum += 1
			self.inpsize = self.result.get_shape().as_list()
			self.activate(activation)
		return [self.result,list(self.inpsize)]


	def dwconvLayer(self,kernel,multi,stride=1,pad='SAME',activation=-1,batch_norm=False,weight=None):
		with tf.variable_scope('dwconv_'+str(self.layernum)):
			if isinstance(kernel,list):
				kernel = kernel
			else:
				kernel = [kernel,kernel]
			self.result = L.conv2Ddw(self.result,self.inpsize[3],kernel,multi,'dwconv_'+str(self.layernum),stride=stride,pad=pad,weight=weight)
			if batch_norm:
				self.result = L.batch_norm(self.result,'batch_norm_'+str(self.layernum))
			self.layernum+=1
			self.inpsize = self.result.get_shape().as_list()
			self.activate(activation)
		return [self.result,list(self.inpsize)]

	def spconvLayer(self,size,multi,stride=1,pad='SAME',activation=-1,batch_norm=False):
		self.dwconvLayer(size,multi,stride=stride,pad=pad)
		self.convLayer(1,self.inpsize[3],activation=activation,batch_norm=batch_norm)
		return [self.result,list(self.inpsize)]

	def deconvLayer(self,kernel,outchn,stride=1,pad='SAME',activation=-1,batch_norm=False):
		self.result = L.deconv2D(self.result,kernel,outchn,'deconv_'+str(self.layernum),stride=stride,pad=pad)
		if batch_norm:
			self.result = L.batch_norm(self.result,'batch_norm_'+str(self.layernum),training=self.bntraining)
		self.layernum+=1
		self.inpsize = self.result.get_shape().as_list()
		self.activate(activation)
		return [self.result,list(self.inpsize)]

	def maxpoolLayer(self,size,pad='SAME',stride=None):
		if stride==None:
			stride = size
		self.result = L.maxpooling(self.result,size,stride,'maxpool_'+str(self.layernum),pad=pad)
		self.inpsize = self.result.get_shape().as_list()
		return [self.result,list(self.inpsize)]

	def avgpoolLayer(self,size,pad='SAME',stride=None):
		if stride==None:
			stride = size
		self.result = L.avgpooling(self.result,size,stride,'maxpool_'+str(self.layernum),pad=pad)
		self.inpsize = self.result.get_shape().as_list()
		return [self.result,list(self.inpsize)]

	def flatten(self):
		self.result = tf.reshape(self.result,[-1,self.inpsize[1]*self.inpsize[2]*self.inpsize[3]])
		self.transShape = [self.inpsize[1],self.inpsize[2],self.inpsize[3],0]
		self.inpsize = [None,self.inpsize[1]*self.inpsize[2]*self.inpsize[3]]
		self.fcs.append(len(self.varlist))
		return [self.result,list(self.inpsize)]

	def construct(self,shape):
		self.result = tf.reshape(self.result,[-1,shape[0],shape[1],shape[2]])
		self.inpsize = [None,shape[0],shape[1],shape[2]]
		return [self.result,list(self.inpsize)]

	def fcLayer(self,outsize,activation=-1,nobias=False,batch_norm=False):
		with tf.variable_scope('fc_'+str(self.layernum)):
			self.inpsize = [i for i in self.inpsize]
			self.result = L.Fcnn(self.result,self.inpsize[1],outsize,'fc_'+str(self.layernum),nobias=nobias)
			if len(self.fcs)!=0:
				if self.fcs[-1] == len(self.varlist):
					self.transShape[-1] = outsize
			self.varlist = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
			if batch_norm:
				self.result = L.batch_norm(self.result,'batch_norm_'+str(self.layernum),training=self.bntraining)
			self.inpsize[1] = outsize
			self.activate(activation)
			self.layernum+=1
		return [self.result,list(self.inpsize)]

	def scale(self,number):
		with tf.variable_scope('scale_'+str(self.layernum)):
			self.result = self.result * number
		return [self.result,list(self.inpsize)]

	def multiply(self,layerin):
		if isinstance(layerin,list):
			self.result = self.result*layerin[0]
		else:
			self.result = self.result*layerin

	def sum(self,layerin):
		assert layerin[1][2] == self.inpsize[2] and layerin[1][1] == self.inpsize[1]
		assert layerin[1][3] == self.inpsize[3]
		with tf.variable_scope('sum_'+str(self.layernum)):
			self.result = self.result +	layerin[0]
		return [self.result,list(self.inpsize)]

	def NIN(self,size,outchn1,outchn2,activation=-1,batch_norm=False,pad='SAME'):
		with tf.variable_scope('NIN_'+str(self.layernum)):
			self.convLayer(1,outchn1,activation=activation,batch_norm=batch_norm)
			self.convLayer(size,outchn2,activation=activation,batch_norm=batch_norm,pad=pad)
		return [self.result,list(self.inpsize)]

	def incep(self,outchn1,outchn2,outchn3,outchn4,outchn5,activation=-1,batch_norm=False):
		with tf.variable_scope('Incep_'+str(self.layernum)):
			orignres = self.result
			orignsize = self.inpsize
			a,_ = self.NIN(3,outchn1,outchn2,activation=activation,batch_norm=batch_norm)
			asize = self.inpsize
			self.inpsize = orignsize
			self.result = orignres
			b,_ = self.NIN(5,outchn3,outchn4,activation=activation,batch_norm=batch_norm)
			bsize = self.inpsize
			self.inpsize = orignsize
			self.result = orignres
			c,_ = self.convLayer(1,outchn5,activation=activation,batch_norm=batch_norm)
			csize = self.inpsize
			self.inpsize[3] = asize[3]+bsize[3]+csize[3]
			self.result = tf.concat(axis=3,values=[a,b,c])
			return [self.result,list(self.inpsize)]

	def concat_to_current(self,layerinfo,axis=3):
		with tf.variable_scope('concat'+str(self.layernum)):
			layerin, layersize = layerinfo[0],list(layerinfo[1])
			self.result = tf.concat(axis=axis,values=[self.result,layerin])
			self.inpsize[axis] += layersize[axis]
		return [self.result,list(self.inpsize)]

	def set_current(self,layerinfo):
		layerin, layersize = layerinfo[0],layerinfo[1]
		self.result = layerin
		self.inpsize = layersize

	def dropout(self,ratio):
		with tf.name_scope('dropout'+str(self.layernum)):
			self.result = tf.nn.dropout(self.result,ratio)
		return [self.result,list(self.inpsize)]

	def l2norm(self,axis=1):
		with tf.name_scope('l2norm'+str(self.layernum)):
			self.result = tf.nn.l2_normalize(self.result,axis)
		return [self.result,list(self.inpsize)]

	def batch_norm(self):
		with tf.variable_scope('batch_norm'+str(self.layernum)):
			self.result = L.batch_norm(self.result,'batch_norm_'+str(self.layernum),training=self.bntraining)
		return [self.result,list(self.inpsize)]

	def resize_nn(self,multip):
		assert self.inpsize[1] == self.inpsize[2]
		with tf.variable_scope('resize_'+str(self.layernum)):
			self.result = L.resize_nn(self.result,multip*self.inpsize[1],name='resize_nn_'+str(self.layernum))
			self.inpsize[1] *= multip
			self.inpsize[2] *= multip
		return [self.result,list(self.inpsize)]

	def reshape(self,shape):
		with tf.variable_scope('reshape_'+str(self.layernum)):
			self.result = tf.reshape(self.result,shape)
			self.inpsize = shape
		return [self.result,list(self.inpsize)]

	def transpose(self,order):
		with tf.variable_scope('transpose_'+str(self.layernum)):
			self.result=tf.transpose(self.result,order)
			self.inpsize = [self.inpsize[i] for i in order]
		return [self.result,list(self.inpsize)]

	def gradient_flip_layer(self):
		with tf.variable_scope('Gradient_flip_'+str(self.layernum)):
			@tf.RegisterGradient("GradFlip")
			def _flip_grad(op,grad):
				return [tf.negative(grad)]

			g = tf.get_default_graph()
			with g.gradient_override_map({'Identity':'GradFlip'}):
				self.result = tf.identity(self.result)
		return [self.result,list(self.inpsize)]

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
		return [self.result,list(self.inpsize)]

	def primaryCaps(self, size, vec_dim, n_chn,activation=None, stride=1,pad='SAME'):
		with tf.variable_scope('Caps_'+str(self.layernum)):
			self.convLayer(size, vec_dim*n_chn, activation=activation, stride=stride, pad=pad)
			shape = self.result.get_shape().as_list()
			self.result = tf.reshape(self.result, [-1,shape[1]*shape[2]*shape[3]//vec_dim,1,vec_dim,1])
			self.inpsize = self.result.get_shape().as_list()
			self.squash()
		return [self.result,list(self.inpsize)]

	def squash(self):
		with tf.variable_scope('squash_'+str(self.layernum)):
			sqr = tf.reduce_sum(tf.square(self.result),-2,keep_dims=True)
			activate = sqr / (1+sqr)
			self.result = activate * tf.nn.l2_normalize(self.result,-2)
		return [self.result,list(self.inpsize)]

	def capsLayer(self,outchn,vdim2,iter_num,BSIZE):
		with tf.variable_scope('capLayer_'+str(self.layernum)):
			# input size: [BSIZE, capin, 1, vdim1,1]
			_,capin,_,vdim1,_ = self.inpsize
			W = L.weight([1,capin,outchn,vdim1,vdim2])
			W = tf.tile(W,[BSIZE,1,1,1,1])
			b = tf.constant(0,dtype=tf.float32,shape=[BSIZE,capin,outchn,1,1])
			res_tile = tf.tile(self.result,[1,1,outchn,1,1])
			# print('W')
			# print(W)
			# print('Res')
			# print(res_tile)
			# input()
			res = tf.matmul(W,res_tile,transpose_a=True)  # [BSIZE, capin, capout, vdim2, 1]
			for i in range(iter_num):
				with tf.variable_scope('Routing_'+str(self.layernum)+'_'+str(i)):
					c = tf.nn.softmax(b,dim=2)
					self.result = tf.reduce_sum(c*res,1,keep_dims=True)  # [BSIZE, 1, capout, vdim2, 1]
					self.squash()
					if i!=iter_num-1:
						b += tf.reduce_sum(self.result * res, -2, keep_dims=True)
			self.result = tf.einsum('ijklm->ikjlm',self.result)
			self.inpsize = [None,outchn,1,vdim2,1]
			self.layernum += 1
		return [self.result,list(self.inpsize)]

	def capsDown(self):
		with tf.variable_scope('Caps_Dim_Down_'+str(self.layernum)):
			self.result = tf.reduce_sum(self.result,-1)
			self.result = tf.reduce_sum(self.result,-2)
			self.inpsize = [None,self.inpsize[1],self.inpsize[3]]
		return [self.result,list(self.inpsize)]

	def capsMask(self,labholder):
		with tf.variable_scope('capsMask_'+str(self.layernum)):
			labholder = tf.expand_dims(labholder,-1)
			self.result = self.result * labholder
			self.result = tf.reshape(self.result,[-1,self.inpsize[1]*self.inpsize[2]])
			self.inpsize = [None,self.inpsize[1]*self.inpsize[2]]
		return [self.result,list(self.inpsize)]
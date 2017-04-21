import layers as L 
import tensorflow as tf

crsentpy = -1
counter = 0

PARAM_RELU = 0
PARAM_LRELU = 1
PARAM_ELU = 2
PARAM_TANH = 3
PARAM_MFM_CONV = 4
PARAM_MFM_FC = 5
PARAM_SIGMOID = 6

def loadSess(modelpath,sess=None,modpath=None,mods=None):
	global counter
	# load session if there exist any models, and initialize the sess if not
	assert modpath==None or mods==None
	if sess==None:
		sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	counter = 0
	saver = tf.train.Saver()
	ckpt = tf.train.get_checkpoint_state(modelpath)
	if modpath!=None:
		mod = modpath
		print('loading from model:',mod)
		saver.restore(sess,mod)
	elif mods!=None:
		for m in mods:
			print('loading from model:',m)
			saver.restore(sess,m)
	elif ckpt:
		mod = ckpt.model_checkpoint_path
		print('loading from model:',mod)
		counter = int(mod.split('/')[-1].replace('ModelCounter','').replace('.ckpt',''))
		saver.restore(sess,mod)
	else:
		print('No checkpoint in folder, use initial graph...')
	return sess

def sparse_softmax_crs_entropy(inp,lab):
	crsentpy+=1
	return L.sparse_softmax_crs_entropy(inp,lab,'cross_entropy_'+str(crsentpy))

class Model():
	def __init__(self,inp,size):
		self.result = inp
		self.inpsize = list(size)
		self.layernum = 0
		self.transShape = None
		self.varlist = []
		self.fcs = []

	def get_current_layer(self):
		return self.result

	def get_shape(self):
		return self.inpsize

	def activate(self,inp,param):
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
				res =  L.MFM(inp,name='mfm_'+str(self.layernum))
			elif param == 5:
				self.inpsize[-1] = self.inpsize[-1]//2
				res =  L.MFMfc(inp,name='mfm_'+str(self.layernum))
			elif param == 6:
				res =  L.sigmoid(inp,name='sigmoid_'+str(self.layernum))
			else:
				res =  inp
		return res

	def convLayer(self,size,outchn,stride=1,pad='SAME',activation=-1,batch_norm=False):
		self.result = L.conv2D(self.result,size,outchn,'conv_'+str(self.layernum),stride=stride,pad=pad)
		self.varlist = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
		# print(self.varlist)
		if batch_norm:
			self.result = L.batch_norm(self.result,'batch_norm_'+str(self.layernum))
		self.result = self.activate(self.result,activation)
		self.layernum += 1
		self.inpsize[1] = self.inpsize[1]//stride
		self.inpsize[2] = self.inpsize[2]//stride
		self.inpsize[3] = outchn
		return [self.result,list(self.inpsize)]

	def deconvLayer(self,size,outchn,stride=1,pad='SAME',activation=-1,batch_norm=False):
		self.result = L.deconv2D(self.result,size,outchn,'deconv_'+str(self.layernum),stride=stride,pad=pad)
		if batch_norm:
			self.result = L.batch_norm(self.result,'batch_norm_'+str(self.layernum))
		self.result = self.activate(self.result,activation)
		self.layernum+=1
		self.inpsize[1] *= stride
		self.inpsize[2] *= stride
		self.inpsize[3] = outchn
		return [self.result,list(self.inpsize)]

	def maxpoolLayer(self,size):
		self.result = L.maxpooling(self.result,size,'maxpool_'+str(self.layernum))
		# print(self.inpsize)
		self.inpsize[1] = int((self.inpsize[1]+0.5)/size)
		self.inpsize[2] = int((self.inpsize[2]+0.5)/size)
		return [self.result,list(self.inpsize)]

	def avgpoolLayer(self,size):
		self.result = L.avgpooling(self.result,size,'maxpool_'+str(self.layernum))
		self.inpsize[1] = self.inpsize[1]//size
		self.inpsize[2] = self.inpsize[2]//size
		return [self.result,list(self.inpsize)]

	def flatten(self):
		self.result = tf.reshape(self.result,[-1,self.inpsize[1]*self.inpsize[2]*self.inpsize[3]])
		self.transShape = [self.inpsize[1],self.inpsize[2],self.inpsize[3],0]
		self.inpsize = [None,self.inpsize[1]*self.inpsize[2]*self.inpsize[3]]
		self.fcs.append(len(self.varlist))
		# print(self.transShape)
		return [self.result,list(self.inpsize)]

	def construct(self,shape):
		self.result = tf.reshape(self.result,[-1,shape[0],shape[1],shape[2]])
		self.inpsize = [None,shape[0],shape[1],shape[2]]
		return [self.result,list(self.inpsize)]

	def fcLayer(self,outsize,activation=-1,nobias=False,batch_norm=False):
		self.result = L.Fcnn(self.result,self.inpsize[1],outsize,'fc_'+str(self.layernum),nobias=nobias)
		if len(self.fcs)!=0:
			if self.fcs[-1] == len(self.varlist):
				self.transShape[-1] = outsize
				# print(self.transShape)
		self.varlist = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
		if batch_norm:
			self.result = L.batch_norm(self.result,'batch_norm_'+str(self.layernum))
		self.inpsize[1] = outsize
		self.result = self.activate(self.result,activation)
		self.layernum+=1
		return [self.result,list(self.inpsize)]

	def NIN(self,size,outchn1,outchn2,activation=-1,batch_norm=False):
		self.convLayer(1,outchn1,activation=activation,batch_norm=batch_norm)
		self.convLayer(size,outchn2,activation=activation,batch_norm=batch_norm)
		return [self.result,list(self.inpsize)]

	def incep(self,outchn1,outchn2,outchn3,outchn4,outchn5,activation=-1,batch_norm=False):
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

	def concat_to_current(self,layerinfo):
		layerin, layersize = layerinfo[0],layerinfo[1]
		# print(layerin.get_shape())
		# print(self.inpsize)
		assert layersize[0] == self.inpsize[0] and layersize[1]==self.inpsize[1]
		self.result = tf.concat(axis=3,values=[self.result,layerin])
		self.inpsize[3] += layersize[3]
		return [self.result,list(self.inpsize)]

	def setSourceLayer(self,layerinfo):
		layerin, layersize = layerinfo[0],layerinfo[1]
		self.result = layerin
		self.inpsize = layersize

	def dropout(self,ratio):
		with tf.name_scope('dropout'+str(self.layernum)):
			self.result = tf.nn.dropout(self.result,ratio)
		return [self.result,list(self.inpsize)]

	def l2norm(self):
		with tf.name_scope('l2norm'+str(self.layernum)):
			self.result = tf.nn.l2_normalize(self.result,1)
		return [self.result,list(self.inpsize)]

	def convertVariablesToCaffe(self,sess,h5name):
		import caffeconverter as cc
		import scipy.io as sio 
		print('varlist:',len(self.varlist))
		f = open('layers.txt')
		dt = {}
		layers = []
		for line in f:
			layers.append(line.replace('\n',''))
		f.close()
		print('layers:',len(layers))
		print('variables:',len(self.varlist))
		for i in range(len(layers)):
			if i*2 in self.fcs:
				print('reshape fc layer...')
				dt[layers[i]+'w'] = cc.reshapeFcWeight(self.varlist[i*2],self.transShape,sess)
			else:
				dt[layers[i]+'w'] = sess.run(self.varlist[i*2])
			dt[layers[i]+'b'] = sess.run(self.varlist[i*2+1])
		sio.savemat('tfModelVars.mat',dt)
		cvt = cc.h5converter(h5name)
		cvt.startConvert()
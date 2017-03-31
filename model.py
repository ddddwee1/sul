import layers as L 
import tensorflow as tf


PARAM_RELU = 0
PARAM_LRELU = 1
PARAM_ELU = 2
PARAM_TANH = 3
PARAM_MFM_CONV = 4
PARAM_MFM_FC = 5

def loadSess(modelpath,sess=None,modpath=None):
#load session if there exist any models, and initialize the sess if not
	if sess==None:
		sess = tf.Session()
	saver = tf.train.Saver()
	ckpt = tf.train.get_checkpoint_state(modelpath)
	if modpath!=None:
		mod = modpath
		print('loading from model:',mod)
		saver.restore(sess,mod)
	elif ckpt:
		mod = ckpt.model_checkpoint_path
		print('loading from model:',mod)
		saver.restore(sess,mod)
	else:
		print('No checkpoint in folder, initializing graph...')
		sess.run(tf.global_variables_initializer())
	return sess

class Model():
	def __init__(self,inp,size):
		self.result = inp
		self.inpsize = list(size)
		self.layernum = 0

	def get_current_layer(self):
		return self.result

	def get_shape(self):
		return self.inpsize

	def activate(self,inp,param):
		if param == 0:
			return L.relu(inp,name='relu_'+str(self.layernum))
		elif param == 1:
			return L.lrelu(inp,name='lrelu_'+str(self.layernum))
		elif param == 2:
			return L.elu(inp,name='elu_'+str(self.layernum))
		elif param == 3:
			return L.tanh(inp,name='tanh_'+str(self.layernum))
		elif param == 4:
			return L.MFM(inp,name='mfm_'+str(self.layernum))
		elif param == 5:
			return L.MFMfc(inp,name='mfm_'+str(self.layernum))
		else:
			return inp

	def convLayer(self,size,outchn,stride=1,pad='SAME',activation=-1,batch_norm=False):
		self.result = L.conv2D(self.result,size,outchn,'conv_'+str(self.layernum),stride=stride,pad=pad)
		if batch_norm:
			self.result = L.batch_norm(self.result,'batch_norm_'+str(self.layernum))
		self.result = self.activate(self.result,activation)
		self.layernum += 1
		self.inpsize[1] = self.inpsize[1]//stride
		self.inpsize[2] = self.inpsize[2]//stride
		self.inpsize[3] = outchn
		return [self.result,self.inpsize]

	def deconvLayer(self,size,outchn,stride=1,pad='SAME',activation=-1,batch_norm=False):
		self.result = L.deconv2D(self.result,size,outchn,'deconv_'+str(self.layernum),stride=stride,pad=pad)
		if batch_norm:
			self.result = L.batch_norm(self.result,'batch_norm_'+str(self.layernum))
		self.result = self.activate(self.result,activation)
		self.layernum+=1
		self.inpsize[1] *= stride
		self.inpsize[2] *= stride
		self.inpsize[3] = outchn
		return [self.result,self.inpsize]

	def maxpoolLayer(self,size):
		self.result = L.maxpooling(self.result,size,'maxpool_'+str(self.layernum))
		self.inpsize[1] = self.inpsize[1]//size
		self.inpsize[2] = self.inpsize[2]//size
		return [self.result,self.inpsize]

	def avgpoolLayer(self,size):
		self.result = L.avgpooling(self.result,size,'maxpool_'+str(self.layernum))
		self.inpsize[1] = self.inpsize[1]//size
		self.inpsize[2] = self.inpsize[2]//size
		return [self.result,self.inpsize]

	def flatten(self):
		self.result = tf.reshape(self.result,[-1,self.inpsize[1]*self.inpsize[2]*self.inpsize[3]])
		self.inpsize = [None,self.inpsize[1]*self.inpsize[2]*self.inpsize[3]]
		return [self.result,self.inpsize]

	def construct(self,shape):
		self.result = tf.reshape(self.result,[-1,shape[0],shape[1],shape[2]])
		self.inpsize = [None,shape[0],shape[1],shape[2]]
		return [self.result,self.inpsize]

	def fcLayer(self,outsize,activation=-1,nobias=False,batch_norm=False):
		self.result = L.Fcnn(self.result,self.inpsize[1],outsize,'fc_'+str(self.layernum),nobias=nobias)
		if batch_norm:
			self.result = L.batch_norm(self.result,'batch_norm_'+str(self.layernum))
		self.result = self.activate(self.result,activation)
		self.layernum+=1
		self.inpsize[1] = outsize
		return [self.result,self.inpsize]

	def NIN(self,size,outchn1,outchn2,activation=-1,batch_norm=False):
		convLayer(1,outchn1,activation=activation,batch_norm=batch_norm)
		convLayer(size,outchn2,activation=activation,batch_norm=batch_norm)
		return [self.result,self.inpsize]

	def incep(self,outchn1,outchn2,outchn3,outchn4,outchn5,activation=-1,batch_norm=False):
		orignsize = self.inpsize
		a = NIN(3,outchn1,outchn2,activation=activation,batch_norm=batch_norm)
		asize = self.inpsize
		self.inpsize = orignsize
		b = NIN(5,outchn3,outchn4,activation=activation,batch_norm=batch_norm)
		bsize = self.inpsize
		self.inpsize = orignsize
		c = convLayer(1,outchn5,activation=activation,batch_norm=batch_norm)
		csize = self.inpsize
		self.inpsize[3] = asize[3]+bsize[3]+csize[3]
		self.result = tf.concat(axis=3,values=[a,b,c])
		return [self.result,self.inpsize]

	def concatToCurrent(self,layerinfo):
		layerin, layersize = layerinfo[0],layerinfo[1]
		assert layersize[0] == self.inpsize[0] and layersize[1]==inpsize[1]
		self.result = tf.concat(axis=3,values=[self.result,layerin])
		self.inpsize[3] += layersize[3]
		return [self.result,self.inpsize]
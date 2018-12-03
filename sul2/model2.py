import layers2 as L 
import tensorflow as tf 
import numpy as np 

PARAM_RELU = 0
PARAM_LRELU = 1
PARAM_ELU = 2
PARAM_TANH = 3
PARAM_MFM = 4
PARAM_MFM_FC = 5
PARAM_SIGMOID = 6

######## util functions ###########
def loadSess(modelpath=None,sess=None,var_list=None,init=False, init_dict=None):
	# load session if there exist any models, and initialize the sess if not
	if init:
		if not modelpath is None:
			if not os.path.exists(modelpath):
				os.makedirs(modelpath)
		print('Initializing...')
		sess.run(tf.global_variables_initializer(),feed_dict=init_dict)
	
	if not modelpath is None:
		saver = tf.train.Saver(var_list)
		ckpt = tf.train.get_checkpoint_state(modelpath)
		if modelpath.endswith('.ckpt'):
			saver.restore(sess,modelpath)
		elif ckpt:
			mod = ckpt.model_checkpoint_path
			print('loading from model:',mod)
			saver.restore(sess,mod)
		else:
			print('No checkpoint in folder, use initial graph...')


########### model class ##########
class Model():
	def __init__(self, inp):
		self.layernum = 0
		self.result = inp
		self.layers = {}
		self.bntraining = True
		self.epsilon = None

	def activate(self,param, **kwarg):
		act = L.activation(self.result, param, 'conv_'+str(self.layernum), kwarg)
		self.result = act.output
		return self.result

	def batch_norm(self):
		bn = L.batch_norm(self.result,training=self.bntraining,epsilon=self.epsilon,'batch_norm_'+str(self.layernum))
		self.result = bn.output
		return self.result

	def convLayer(self,size,outchn,dilation_rate=1,stride=1,pad='SAME',activation=-1,batch_norm=False,layerin=None,usebias=True,kernel_data=None,bias_data=None,weight_norm=False):
		if not layerin is None:
			self.result = layerin

		# conv
		conv = L.conv2D(self.result,kernel,outchn,'conv_'+str(self.layernum),stride=stride,pad=pad,usebias=usebias,kernel_data=kernel_data,bias_data=bias_data,dilation_rate=dilation_rate,weight_norm=weight_norm)
		self.result = conv.output

		# bn
		if batch_norm:
			bn = self.batch_norm()
		# act
		act = self.activate(activation)

		self.layernum += 1
		return self.result

	def maxpoolLayer(self,size,stride=None,pad='SAME'):
		pool = L.maxpoolLayer(self.result,size,stride,'maxpool_'+str(self.layernum),pad=pad)
		self.result = pool.output
		return self.result

	def fcLayer(self,outsize,activation=-1,nobias=False,batch_norm=False):

		fc = L.fcLayer(self.result,outsize,'fc_'+str(self.layernum),nobias=nobias)
		self.result = fc.output
		
		# bn
		if batch_norm:
			self.batch_norm(self.result)
		# act
		self.activate(activation)

		self.layernum+=1
		return self.result

	def flatten(self):
		size = self.result.get_shape().as_list()
		if len(size)==5:
			self.result = tf.reshape(self.result,[-1,size[1]*size[2]*size[3]*size[4]])
		else:
			self.result = tf.reshape(self.result,[-1,size[1]*size[2]*size[3]])
		return self.result

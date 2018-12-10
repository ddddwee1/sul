import tensorflow as tf 
import model2 as M 
import layers2 as L 

class encoder():
	def __init__(self,x):
		self.x = x 

		self.mod = M.Model(x)
		self.depth(x, 1, 32, 1)

	def depth(self, x, blocks, out, depth):
		res = []
		with tf.variable_scope('depth%d'%depth):
			for i in range(blocks):
				self.mod.reset_layernum()
				self.conv_bunch(x, out, stride=2, k=5)
				res.append(self.mod.get_current_layer())
		res = tf.concat(res, axis=-1)
		self.mod.set_current_layer(res)
		return res
			
	def conv_bunch(self,x, out, stride, k=3, activation=M.PARAM_LRELU):
		self.mod.set_current_layer(x)
		self.mod.convLayer(k, out, stride=stride, activation=activation)
		return self.mod.get_current_layer()

	def get_current_layer(self):
		return self.mod.get_current_layer()

class decoder():
	def __init__(self,x):
		self.x = x 

		self.mod = M.Model(x)
		# x = self.depth(x, 1, 32, 2)
		with tf.variable_scope('depth1'):
			self.mod.reset_layernum()
			self.deconv_bunch(x, 3, stride=2, k=5, activation=M.PARAM_TANH)
			
	def depth(self, x, blocks, out, depth):
		res = []
		with tf.variable_scope('depth%d'%depth):
			for i in range(blocks):
				self.mod.reset_layernum()
				self.deconv_bunch(x, out, stride=2, k=5)
				res.append(self.mod.get_current_layer())
		res = tf.concat(res, axis=-1)
		self.mod.set_current_layer(res)
		return res

	def deconv_bunch(self,x, out, stride, k=3, activation=M.PARAM_LRELU):
		self.mod.set_current_layer(x)
		self.mod.deconvLayer(k, out, stride=stride, activation=activation)
		return self.mod.get_current_layer()

	def get_current_layer(self):
		return self.mod.get_current_layer()

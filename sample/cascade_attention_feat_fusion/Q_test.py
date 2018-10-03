import tensorflow as tf 
import model as M 

class network():
	def __init__(self):

		self.inpholder = tf.placeholder(tf.float32,[None,112,112,3])
		self.labholder = tf.placeholder(tf.float32,[None,101])

		# a = tf.map_fn(self.model, self.inpholder)
		feat = self.model(self.inpholder)

		feat = self.q_att(feat)

		print(feat)

	def model(self,inp):
		with tf.variable_scope('modell',reuse=tf.AUTO_REUSE):
			mod = M.Model(inp)
			# print(mod.inpsize)
			mod.flatten()
			mod.fcLayer(256)
		return mod.get_current_layer()

	def q_att(self,feat):
		with tf.variable_scope('Qatt'):
			self.q = tf.get_variable('q_weight',[1,256],initializer=tf.contrib.layers.xavier_initializer())
			mod = M.Model(self.q)
			mod.QAttention(feat)
			mod.fcLayer(256,activation=M.PARAM_TANH)
			mod.QAttention(feat)
		return mod.get_current_layer()


if __name__=='__main__':
	net = network()
	# you can see the output here
	# then you can use the feature for classificcation
	
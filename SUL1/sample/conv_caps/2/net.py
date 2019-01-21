import model as M 
import tensorflow as tf 

class network():
	def __init__(self):
		with tf.variable_scope('holders'):
			self.inp_holder = tf.placeholder(tf.float32,[None,28,28,1])
			self.lab_holder = tf.placeholder(tf.float32,[None,10])

		with tf.variable_scope('mainMod'):
			mod = M.Model(self.inp_holder)
			mod.convLayer(7,64,stride=2,activation=M.PARAM_RELU)
			mod.convLayer(5,128,stride=2,activation=M.PARAM_RELU)
			mod.capsulization(dim=16,caps=8)
			mod.caps_conv(3,8,16,activation=None,usebias=False)
			mod.caps_flatten()
			mod.squash()
			mod.capsLayer(10,8,3,BSIZE=128)
			mod.squash()
			feat = mod.capsDown()

		with tf.variable_scope('loss'):
			length = tf.norm(feat, axis=2)
			self.length = length
			loss = self.lab_holder * tf.square(tf.maximum(0., 0.9 - length)) + 0.5 * (1 - self.lab_holder) * tf.square(tf.maximum(0., length - 0.1))
			self.loss = tf.reduce_mean(tf.reduce_sum(loss,1))
			self.accuracy = M.accuracy(length,tf.argmax(self.lab_holder, 1))

		with tf.variable_scope('opti'):
			self.train_op = tf.train.AdamOptimizer(0.001).minimize(self.loss)

		self.sess = tf.Session()
		M.loadSess(self.sess,'./model/',init=True)

	def train(self, img, lab):
		ls, ac, _ = self.sess.run([self.loss, self.accuracy, self.train_op], feed_dict={self.inp_holder:img, self.lab_holder:lab})
		return ls, ac 

	def eval(self, img):
		l = self.sess.run(self.length, feed_dict={self.inp_holder:img})
		return l
import model as M 
import tensorflow as tf 
import numpy as np 

class feat_model():
	def __init__(self,feat_num, class_num):
		self.feat_num = feat_num
		self.class_num = class_num

		self.feat_holder = tf.placeholder(tf.float32,[None,None,feat_num])
		self.lab_holder = tf.placeholder(tf.float32,[None,class_num])

		# simple classifier
		fused_feat = self.Qatt(self.feat_holder)
		self.Classifier(fused_feat)

		self.sess = tf.Session()
		self.saver = tf.train.Saver()
		M.loadSess('./model/',self.sess, init=True)

	def Qatt(self,feat):
		with tf.variable_scope('QATT'):
			q = tf.get_variable('q_weight',[1,self.feat_num],initializer=tf.contrib.layers.xavier_initializer())
			mod = M.Model(q)
			mod.tile([tf.shape(feat)[0],1])
			mod.QAttention(feat)
			mod.fcLayer(self.feat_num,activation=M.PARAM_TANH)
			mod.QAttention(feat)
		return mod.get_current_layer()

	def Classifier(self,feat):
		with tf.variable_scope('classifier'):
			mod = M.Model(feat)
			mod.fcLayer(self.class_num, nobias=True)
			self.scr = mod.get_current_layer()
		with tf.variable_scope('Optimizer'):
			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scr, labels=self.lab_holder))
			self.accuracy = M.accuracy(self.scr, tf.argmax(self.lab_holder, -1))
			self.train_op = tf.train.AdamOptimizer(0.0001).minimize(self.loss)

	def train(self, data,lab):
		data = np.float32(data)
		lab = np.float32(lab)
		ac, ls , _ = self.sess.run([self.accuracy, self.loss, self.train_op], feed_dict = {self.feat_holder:data, self.lab_holder:lab})
		return ac, ls 

	def eval(self, data,lab):
		data = np.float32(data)
		lab = np.float32(lab)
		ac, ls = self.sess.run([self.accuracy, self.loss], feed_dict = {self.feat_holder:data, self.lab_holder:lab})
		return ac, ls 

	def save(self,path):
		self.saver.save(self.sess, './model/%s.ckpt'%path)

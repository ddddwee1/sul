import model as M 
import tensorflow as tf 
import numpy as np 


class res_101():
	def __init__(self,class_num,is_training=True,mod_dir='./model/'):
		self.mod_dir = mod_dir
		with tf.variable_scope('Input'):
			self.img_holder = tf.placeholder(tf.float32,[None,128,128,3])
			self.lab_holder = tf.placeholder(tf.float32,[None,class_num])
		with tf.variable_scope('Res_101_cy'):
			mod = M.Model(self.img_holder)
			mod.set_bn_training(is_training)
			# 64x64
			mod.convLayer(7,64,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
			mod.res_block(256,stride=1,activation=M.PARAM_LRELU)
			mod.res_block(256,stride=1,activation=M.PARAM_LRELU)
			mod.res_block(256,stride=1,activation=M.PARAM_LRELU)
			# 32x32
			mod.res_block(512,stride=2,activation=M.PARAM_LRELU)
			mod.res_block(512,stride=1,activation=M.PARAM_LRELU)
			mod.res_block(512,stride=1,activation=M.PARAM_LRELU)
			mod.res_block(512,stride=1,activation=M.PARAM_LRELU)
			# 16x16
			mod.res_block(1024,stride=1,activation=M.PARAM_LRELU)
			for i in range(13):
				mod.res_block(1024,stride=1,activation=M.PARAM_LRELU)
			# 8x8
			mod.res_block(2048,stride=2,activation=M.PARAM_LRELU)
			mod.res_block(2048,stride=1,activation=M.PARAM_LRELU)
			mod.res_block(2048,stride=1,activation=M.PARAM_LRELU)
			mod.avgpoolLayer(8)
			mod.flatten()
			#mod.fcLayer(256,nobias=True)
		self.feat = mod.get_current_layer()
		with tf.variable_scope('Classification'):
			logit_layer, eval_layer = M.enforcedClassifier(self.feat, self.lab_holder, dropout=1, multi=None, L2norm=False)
			self.accuracy = M.accuracy(eval_layer, tf.argmax(self.lab_holder,-1))

		if is_training:
			print('Building optimizer...')
			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit_layer, labels=self.lab_holder))
			with tf.control_dependencies(M.get_update_ops()):
				self.train_op = tf.train.AdamOptimizer(0.0001).minimize(self.loss)

		self.sess = tf.Session()
		self.saver = tf.train.Saver()
		M.loadSess(mod_dir, self.sess, init=True)

	def train(self, img, lab, normalize=True):
		img = np.float32(img)
		lab = np.float32(lab)
		if normalize:
			img = img / 127.5 - 1. 
		ls,acc, _ = self.sess.run([self.loss, self.accuracy, self.train_op], \
			feed_dict={self.img_holder:img, self.lab_holder:lab})
		return ls,acc

	def eval(self,img,lab,normalize=True):
		img = np.float32(img)
		if normalize:
			img = img/127.5 - 1.
		acc = self.sess.run(self.accuracy,feed_dict={self.img_holder:img, self.lab_holder:lab})
		return acc

	def get_feat(self,img,normalize=True):
		img = np.float32(img)
		if normalize:
			img = img/127.5 - 1.
		feat = self.sess.run(self.feat,feed_dict={self.img_holder:img})
		return feat

	def save(self,fname):
		print('Saving to file:', self.mod_dir+fname)
		self.saver.save(self.sess, self.mod_dir+fname)

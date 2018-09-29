import tensorflow as tf 
import numpy as np 
import model as M 


class video_conv3d():
	def __init__(self,classnum,isTraining=True, model_path='./model/'):
		self.model_path = model_path
		# create input placeholder and label placeholder
		self.input_holder = tf.placeholder(tf.float32,[None,8,224,224,3])
		self.lab_holder = tf.placeholder(tf.float32,[None,classnum])

		# build model and classifier and optimizer
		self.feat = self.model()
		self.build_classifier()

		# create session and saver
		self.sess = tf.Session()
		self.saver = tf.train.Saver()
		M.loadSess(self.model_path,self.sess,init=True)

	def model(self):
		with tf.variable_scope('conv3d_incep'):
			mod = M.Model(self.input_holder)
			self.blk_num = 0
			mod.conv3dLayer(7,64,activation=M.PARAM_LRELU,stride=[1,2,2]) # 112x112
			mod.maxpool3dLayer([1,3,3],stride=[1,2,2]) # 8x56x56
			mod.conv3dLayer(1,64,activation=M.PARAM_LRELU)
			mod.conv3dLayer(3,192,activation=M.PARAM_LRELU)
			mod.maxpool3dLayer([2,3,3],stride=[2,2,2]) # 8x28x28

			self.inc(mod,[64,96,128,16,32,32]) #256
			self.inc(mod,[96,128,192,32,48,48]) # 384
			mod.maxpool3dLayer(3,stride=[1,2,2]) # 4x14x14

			for i in range(5):
				self.inc(mod,[128,128,256,32,64,64]) # 512

			mod.maxpool3dLayer(2,stride=2) # 2x7x7
			self.inc(mod,[256,256,512,64,128,128])
			self.inc(mod,[256,256,512,64,128,128])
			mod.avgpool3dLayer([2,7,7],pad='VALID') # 1x1x1

			mod.flatten()
		return mod.get_current_layer()

	def inc(self,mod, feature_map_num):
		with tf.variable_scope('IncBlock_%d'%self.blk_num):
			self.blk_num += 1
			input_layer = mod.get_current_layer()
			branch_1x1 = mod.conv3dLayer(1, feature_map_num[0],activation=M.PARAM_LRELU,batch_norm=True, layerin=input_layer)
			mod.conv3dLayer(1, feature_map_num[1],activation=M.PARAM_LRELU,batch_norm=True, layerin=input_layer)
			branch_3x3 = mod.conv3dLayer(3, feature_map_num[2],activation=M.PARAM_LRELU,batch_norm=True)
			mod.conv3dLayer(1, feature_map_num[3],activation=M.PARAM_LRELU,batch_norm=True, layerin=input_layer)
			branch_5x5 = mod.conv3dLayer(5, feature_map_num[4],activation=M.PARAM_LRELU,batch_norm=True)
			mod.set_current_layer(input_layer)
			mod.maxpool3dLayer(3,stride=1)
			branch_pool = mod.conv3dLayer(1, feature_map_num[5],activation=M.PARAM_LRELU,batch_norm=True)
			concat_layer = tf.concat([branch_1x1, branch_3x3, branch_5x5, branch_pool], axis=-1)
			mod.set_current_layer(concat_layer)

	def build_classifier(self):
		with tf.variable_scope('classifier'):
			logit_layer, eval_layer = M.enforcedClassifier(self.feat, self.lab_holder, dropout=1., multi=None)
			self.accuracy = M.accuracy(eval_layer, tf.argmax(self.lab_holder,-1))
		self.eval_layer = eval_layer
		with tf.variable_scope('optimizer'):
			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit_layer, labels=self.lab_holder))
			with tf.control_dependencies(M.get_update_ops()):
				self.train_step = tf.train.AdamOptimizer(0.0001).minimize(logit_layer)

	def train(self, inp, lab, normalize=True):
		inp = np.float32(inp)
		lab = np.float32(lab)
		if normalize:
			inp = inp / 127.5 - 1.
		ls,acc, _ = self.sess.run([self.loss, self.accuracy, self.train_step], feed_dict = {self.input_holder:inp, self.lab_holder:lab})
		return ls,acc 

	def get_score(self, inp, normalize=True):
		inp = np.float32(inp)
		if normalize:
			inp = inp / 127.5 - 1.
		scr = self.sess.run(eval_layer, feed_dict = {self.input_holder:inp, self.lab_holder:lab})
		return scr 

	def save(self, name):
		print('Saving model to',self.model_path+name,'...')
		self.saver.save(self.sess, self.model_path+name)

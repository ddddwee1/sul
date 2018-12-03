import tensorflow as tf 
import numpy as np 
import model as M 

class video_conv3d():
	def __init__(self,classnum, accum=1,isTraining=True, model_path='./model/'):
		self.accumulation = accum
		self.classnum = classnum
		self.model_path = model_path
		self.global_step = 0
		# create input placeholder and label placeholder
		self.input_holder = tf.placeholder(tf.float32,[None,None,16,112,112,3])
		self.lab_holder = tf.placeholder(tf.float32,[None,classnum])
		self.dropout = tf.placeholder(tf.float32)

		# build model and classifier and optimizer
		feat = tf.map_fn(self.model,self.input_holder)
		self.feat = self.feat_fusion(feat)

		self.build_classifier()

		# create session and saver
		self.sess = tf.Session()
		self.saver = tf.train.Saver()
		M.loadSess(self.model_path,self.sess,init=True)

	def model(self,inp):
		with tf.variable_scope('conv3d_incep',reuse=tf.AUTO_REUSE):
			mod = M.Model(inp)
			self.blk_num = 0
			mod.conv3dLayer(3,64,activation=M.PARAM_LRELU) 
			mod.maxpool3dLayer([1,2,2],stride=[1,2,2]) # 56
			mod.conv3dLayer(3,128,activation=M.PARAM_LRELU)
			mod.maxpool3dLayer(2) # 28
			mod.conv3dLayer(3,256,activation=M.PARAM_LRELU)
			mod.conv3dLayer(3,256,activation=M.PARAM_LRELU)
			mod.maxpool3dLayer(2) # 14
			mod.conv3dLayer(3,512,activation=M.PARAM_LRELU)
			mod.conv3dLayer(3,512,activation=M.PARAM_LRELU)
			mod.maxpool3dLayer(2) # 7
			mod.conv3dLayer(3,512,activation=M.PARAM_LRELU)
			mod.conv3dLayer(3,512,activation=M.PARAM_LRELU)
			mod.maxpool3dLayer(2) # 4
			print(mod.get_current_layer())
			mod.flatten()
			mod.fcLayer(2048,activation=M.PARAM_LRELU)
			mod.dropout(self.dropout)
			mod.fcLayer(1024,activation=M.PARAM_LRELU)
			# mod.dropout(self.dropout)

		return mod.get_current_layer()

	def feat_fusion(self,feats):
		with tf.variable_scope('fusion'):
			mod = M.Model(feats)
			mod.dyn_route(3)
		return mod.get_current_layer()

	def build_classifier(self):
		with tf.variable_scope('classifier'):
			logit_layer, eval_layer = M.enforcedClassifier(self.feat, self.lab_holder, dropout=1., multi=None)
			# logit_layer = eval_layer = self.feat
			self.accuracy = M.accuracy(eval_layer, tf.argmax(self.lab_holder,-1))
		self.eval_layer = eval_layer
		with tf.variable_scope('optimizer'):
			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit_layer, labels=self.lab_holder))
			with tf.control_dependencies(M.get_update_ops()):
				trainer = M.Trainer(0.0001, self.loss)
				self.train_step = trainer.train()
				self.accum_step = trainer.accumulate()

	def train(self, inp, lab, normalize=True):
		self.global_step += 1
		inp = np.float32(inp)
		lab = np.float32(lab)
		if normalize:
			inp = inp / 127.5 - 1.
		train_step = self.train_step if self.global_step%self.accumulation==0 else self.accum_step
		ls,acc, _ = self.sess.run([self.loss, self.accuracy, train_step], feed_dict = {self.input_holder:inp, self.lab_holder:lab,\
									self.dropout:0.5})
		return ls,acc 

	def eval(self, inp, lab, normalize=True):
		inp = np.float32(inp)
		lab = np.float32(lab)
		if normalize:
			inp = inp / 127.5 - 1.
		ls,acc = self.sess.run([self.loss, self.accuracy], feed_dict = {self.input_holder:inp, self.lab_holder:lab, self.dropout:1.0})
		return ls,acc 

	def get_score(self, inp, normalize=True):
		inp = np.float32(inp)
		if normalize:
			inp = inp / 127.5 - 1.
		scr = self.sess.run(self.eval_layer, feed_dict = {self.input_holder:inp, self.dropout:1.0})
		return scr 

	def get_feature(self, inp, normalize=True):
		inp = np.float32(inp)
		if normalize:
			inp = inp / 127.5 - 1.
		feat = self.sess.run(self.feat, feed_dict = {self.input_holder:inp, self.dropout:1.0})
		return feat 

	def save(self, name):
		print('Saving model to',self.model_path+name,'...')
		self.saver.save(self.sess, self.model_path+name)

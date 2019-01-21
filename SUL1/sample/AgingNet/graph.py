import tensorflow as tf 
import dyn.model_attention as M 
import network as N 
import numpy as np 

class AgingNet():
	def __init__(self,training=True):
		N.bn_training = training
		self.inpholder = tf.placeholder(tf.float32,[None,128,128,3])
		self.feat = N.feat_encoder(self.inpholder)

		self.age_features = []

		self.recon_features = []
		self.recon_imgs = []

		self.dis_scores = []
		self.dis_feat_scores = []
		self.dis_scores_real = []
		self.dis_feat_scores_real = []

		for i in range(10):
			age_feat = N.age_encoder(self.feat,i)
			self.age_features.append(age_feat)

			recon_img = N.generator(age_feat)
			self.recon_imgs.append(recon_img)

			recon_feature = N.feat_encoder(recon_img)
			self.recon_features.append(recon_feature)

			dis_scr = N.discriminator(recon_img)
			self.dis_scores.append(dis_scr)

			feat_dis_scr = N.discriminator_feature(recon_feature)
			self.dis_feat_scores.append(feat_dis_scr)

			dis_scr_real = N.discriminator(self.inpholder)
			self.dis_scores_real.append(dis_scr_real)

			feat_dis_scr_real = N.discriminator_feature(self.feat)
			self.dis_feat_scores_real.append(feat_dis_scr_real)

		self.age_features = tf.stack(self.age_features,1)
		self.img_feature = N.attention_blk(self.age_features)

	def load_model(self,path='./model/'):
		self.sess = tf.Session()
		M.loadSess(path,self.sess)

	def eval(self,data,normalize=True):
		if normalize:
			data = np.float32(data) / 127.5 - 1.
		else:
			data = np.float32(data)
		if len(data.shape)==3:
			data = [data]
		if not self.sess:
			self.load_model()
		res = self.sess.run(self.img_feature,feed_dict={self.inpholder:data})
		return res

def siamese_aging_net():
	def __init__(self):
		# build nets
		left_net = AgingNet()
		right_net = AgingNet()

		# build losses
		left_dis = self.dis_loss(left_net.dis_scores,False)
		left_dis_feat = self.dis_loss(left_net.dis_feat_scores,False)
		left_dis2 = self.dis_loss(left_net.dis_scores,True)
		left_dis_feat2 = self.dis_loss(left_net.dis_feat_scores,True)
		left_dis_real = self.dis_loss(left_net.dis_scores_real,True)
		left_dis_feat_real = self.dis_loss(left_net.dis_feat_scores_real,True)

		right_dis = self.dis_loss(right_net.dis_scores,False)
		right_dis_feat = self.dis_loss(right_net.dis_feat_scores,False)
		right_dis2 = self.dis_loss(right_net.dis_scores,True)
		right_dis_feat2 = self.dis_loss(right_net.dis_feat_scores,True)
		right_dis_real = self.dis_loss(right_net.dis_scores_real,True)
		right_dis_feat_real = self.dis_loss(right_net.dis_feat_scores_real,True)

		labelholder = tf.placeholder(tf.float32,[None])
		contra_loss = self.contra_loss(left_net.img_feature,right_net.img_feature,0.3, labelholder)

		target_left = tf.placeholder(tf.float32,[None,128,128,3])
		target_right = tf.placeholder(tf.float32,[None,128,128,3])

		recon_loss_left = self.recon_loss(left_net.recon_imgs, target_left)
		recon_loss_right = self.recon_loss(left_net.recon_imgs, target_right)

		# establish train_op
		self.train_gan_left = self.gan_optimizer(left_dis, left_dis_feat, left_dis_real, left_dis_feat_real, left_dis2, left_dis_feat2, recon_loss_left)
		self.train_gan_right = self.gan_optimizer(right_dis, right_dis_feat, right_dis_real, right_dis_feat_real, right_dis2, right_dis_feat2, recon_loss_right)

		self.train_contra = self.contra_optimizer(contra)

		# set class variables 
		self.labelholder = labelholder
		self.gan_loss = left_dis + left_dis_feat + left_dis2 + left_dis_feat2 + left_dis_real + left_dis_feat_real + right_dis + right_dis_feat + right_dis2 + right_dis_feat2 + right_dis_real + right_dis_feat_real
		self.contra_loss = contra_loss
		self.recon_loss = recon_loss_left + recon_loss_right
		self.left_net = left_net
		self.right_net = right_net
		self.target_left = target_left
		self.target_right = target_right

	def dis_loss(self,scr,is_real=True):
		res = []
		with tf.variable_scope('dis_loss'):
			for i in range(10):
				if is_real:
					lab = tf.ones_like(scr[i])
				else:
					lab = tf.zeros_like(scr[i])
				loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=scr[i],labels=lab))
				res.append(loss)
		return res 

	def contra_loss(self,feat1,feat2,margin, label):
		with tf.variable_scope('contra_loss'):
			d = tf.reduce_sum(tf.square(feat1 - feat2),1)
			d_ = tf.sqrt(d)
			loss = label * tf.square(tf.maximum(0., margin - d_)) + (1. - label) * d
			loss = tf.reduce_mean(loss)
		return loss

	def recon_loss(self,recon,target):
		res = []
		with tf.variable_scope('recon_loss'):
			for i in range(10):
				# add some noise here
				ls = recon[i] - target
				ls = tf.reduce_sum(tf.reduce_mean(tf.abs(ls),0))
				res.append(ls)
		return res 

	def gan_optimizer(self, dis, dis_feat, dis_real, dis_feat_real, dis2, dis_feat2, recon):
		res = []
		for i in range(10):
			train_op = tf.train.AdamOptimizer(0.0001).minimize(dis[i]+dis_feat[i]+dis_real[i]+dis_feat_real[i],var_list=M.get_trainable_vars('discriminator')+M.get_trainable_vars('discriminator_feature'))
			train_op2 = tf.train.AdamOptimizer(0.0001).minimize(dis2[i]+dis_feat2[i]+recon[i],var_list=M.get_trainable_vars('generator')+M.get_trainable_vars('decoder'+str(i)))
			with tf.control_denpendencies(M.get_update_ops('discriminator')+M.get_update_ops('discriminator_feature')+M.get_update_ops('generator')+M.get_update_ops('decoder'+str(i))+[train_op2, train_op]):
				train_op = tf.no_op()
			res.append(train_op)
		return res

	def contra_optimizer(self, contra):
		decoder_vars = []
		for i in range(10):
			decoder_vars += M.get_trainable_vars('decoder'+str(i))
		train_op = tf.train.AdamOptimizer(0.0001).minimize(contra,var_list=decoder_vars+M.get_trainable_vars('attention_blk'))
		train_op2 = tf.train.AdamOptimizer(1e-6).minimize(contra,var_list=M.get_trainable_vars('encoder'))
		with tf.control_denpendencies([train_op,train_op2]):
			train_op = tf.no_op()
		return train_op

	def load_model(self,path='./model/'):
		self.sess = tf.Session()
		M.loadSess(path,self.sess)

	def train_gan(self,left, left_target, left_index, right, right_target, right_index, normalize=True, return_loss=False):
		if normalize:
			left = np.float32(left) / 127.5 - 1.
			right = np.float32(right) / 127.5 - 1.
			left_target = np.float32(left_target) / 127.5 - 1.
			right_target = np.float32(right_target) / 127.5 - 1.
		else:
			left = np.float32(left)
			right = np.float32(right)
			left_target = np.float32(left_target)
			right_target = np.float32(right_target)
		feed_dict = {self.left_net.inpholder: left, self.target_left: left_target, self.right_net.inpholder: right, self.target_right: right_target}
		self.sess.run(self.train_gan_left[left_index], feed_dict = feed_dict)
		self.sess.run(self.train_gan_right[right_index], feed_dict = feed_dict)
		if return_loss:
			gan_loss, recon_loss = self.sess.run([self.gan_loss, self.recon_loss],feed_dict = feed_dict)
			return gan_loss, recon_loss
		return 0,0

	def train_contra(self, left, right, label, normalize=True, return_loss=False):
		if normalize:
			left = np.float32(left) / 127.5 - 1.
			right = np.float32(right) / 127.5 - 1.
		else:
			left = np.float32(left)
			right = np.float32(right)
		feed_dict = {self.left_net.inpholder: left, self.right_net.inpholder: right, self.labelholder: label}
		self.sess.run(self.train_contra,feed_dict=feed_dict)
		if return_loss:
			contra_loss = self.sess.run(self.contra_loss ,feed_dict=feed_dict)
			return contra_loss
		return 0

	def generate(self, data, normalize=True):
		if normalize:
			data = np.float32(data) / 127.5 - 1.
		else:
			data = np.float32(data)
		if len(data.shape)==3:
			data = [data]
		recon = self.sess.run(self.left_net.recon_imgs, feed_dict={self.left_net.inpholder:data})
		recon = [(i+1.)*127.5 for i in recon]
		recon = [np.uint8(i) for i in recon]
		return recon

	def eval(self, data, normalize=True):
		if normalize:
			data = np.float32(data) / 127.5 - 1.
		else:
			data = np.float32(data)
		if len(data.shape)==3:
			data = [data]
		feat = self.sess.run(self.left_net.img_feature, feed_dict={self.left_net.inpholder:data})
		return feat 
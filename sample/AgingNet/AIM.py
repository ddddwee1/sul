import tensorflow as tf 
import model as M 
import network as N 
import numpy as np 

class AIM():
	def __init__(self,id_num,age_size,model_path = './aim_model/'):
		self.model_path = model_path
		self.inp_holder = tf.placeholder(tf.float32,[None,128,128,3])
		# self.real_holder = tf.placeholder(tf.float32,[None,128,128,3])
		self.uni_holder = tf.placeholder(tf.float32,[None,2,2,512])
		self.age_holder = tf.placeholder(tf.float32,[None,age_size])
		self.target_holder = tf.placeholder(tf.float32,[None,128,128,3])
		self.id_holder = tf.placeholder(tf.float32,[None,id_num])

		# get_feature
		self.feat = N.feat_encoder(self.inp_holder)

		# retrieve tensor for adv1 and ip
		adv1, ip = N.discriminator_f(self.feat, id_num)
		adv1_uni, _ = N.discriminator_f(self.uni_holder, id_num)

		# get attention A and C
		age_expanded = self.expand(self.age_holder, self.feat)
		aged_feature = tf.concat([age_expanded, self.feat],-1)
		self.A, self.C = N.generator_att(aged_feature)

		# construct synthesized image
		self.generated = self.A * self.C + (1-self.A) * self.inp_holder

		# retrieve tensor for adv2 and ae
		adv2, age_pred = N.discriminator(self.generated, age_size)
		adv2_real, age_pred_real = N.discriminator(self.target_holder, age_size)

		# retrieve tensor for ai1 and ai2
		ai1 = N.age_classify_r(self.feat,age_size)
		ai2 = N.age_classify(self.feat,age_size)

		# call loss builder functions
		print('Building losses...')
		self.build_loss_mc()
		self.build_loss_adv1(adv1,adv1_uni)
		self.build_loss_ip(ip)
		self.build_loss_adv2(adv2,adv2_real)
		self.build_loss_ae(age_pred,age_pred_real)
		self.build_loss_ai1(ai1)
		self.build_loss_ai2(ai2,age_size)
		self.build_loss_A()
		self.update_ops()

		self.sess = tf.Session()
		M.loadSess(model_path,self.sess,init=True)
		self.saver = tf.train.Saver()

	def build_loss_mc(self):
		self.mc_loss = tf.reduce_mean(tf.abs(self.generated - self.target_holder))
		train_mc = tf.train.AdamOptimizer(0.0001).minimize(self.mc_loss,\
			var_list=M.get_all_vars('encoder')+M.get_all_vars('gen_att'))
		# opt_mc = M.get_update_ops('encoder')+M.get_update_ops('gen_att')
		with tf.control_dependencies([train_mc]):
			self.train_mc = tf.no_op()

	def build_loss_adv1(self,adv1,adv1_uni):
		adv1_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=adv1,labels=tf.zeros_like(adv1)))
		adv1_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=adv1_uni,labels=tf.ones_like(adv1_uni)))
		self.adv1_loss_d = 0.5 * (adv1_loss1 + adv1_loss2)

		self.adv1_loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=adv1,labels=tf.ones_like(adv1)))

		# opt_adv1 = M.get_update_ops('encoder') + M.get_update_ops('dis_f')

		train_adv1_d = tf.train.AdamOptimizer(0.0001).minimize(self.adv1_loss_d,\
			var_list=M.get_all_vars('dis_f'))
		train_adv1_g = tf.train.AdamOptimizer(0.0001).minimize(self.adv1_loss_g,\
			var_list=M.get_all_vars('encoder'))

		with tf.control_dependencies([train_adv1_g,train_adv1_d]):
			self.train_adv1 = tf.no_op()

	def build_loss_ip(self,ip):
		self.ip_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ip,labels=self.id_holder))

		self.train_ip = tf.train.AdamOptimizer(0.0001).minimize(self.ip_loss)

	def build_loss_adv2(self,adv2,adv2_real):
		adv2_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=adv2,labels=tf.zeros_like(adv2)))
		adv2_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=adv2_real,labels=tf.ones_like(adv2_real)))
		self.adv2_loss_d = 0.5 * (adv2_loss1 + adv2_loss2)

		self.adv2_loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=adv2,labels=tf.ones_like(adv2)))

		train_adv2_d = tf.train.AdamOptimizer(0.0001).minimize(self.adv2_loss_d,\
			var_list=M.get_all_vars('discriminator'))

		train_adv2_g = tf.train.AdamOptimizer(0.0001).minimize(self.adv2_loss_g,\
			var_list=M.get_all_vars('gen_att'))

		with tf.control_dependencies([train_adv2_d,train_adv2_g]):
			self.train_adv2 = tf.no_op()

	def build_loss_ae(self,age_pred, age_pred_real):
		self.age_cls_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=age_pred_real,labels=self.age_holder))
		self.age_generate_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=age_pred,labels=self.age_holder))

		train_classifier = tf.train.AdamOptimizer(0.0001).minimize(self.age_cls_loss,\
			var_list=M.get_all_vars('discriminator'))
		train_generator = tf.train.AdamOptimizer(0.0001).minimize(self.age_generate_loss,\
			var_list=M.get_all_vars('gen_att'))

		with tf.control_dependencies([train_classifier,train_generator]):
			self.train_ae = tf.no_op()

	def build_loss_ai1(self,ai1):
		self.ai1_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ai1,labels=self.age_holder))
		self.train_ai1 = tf.train.AdamOptimizer(0.0001).minimize(self.ai1_loss)

	def build_loss_ai2(self,ai2,age_size):
		self.ai2_cls_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ai2,labels=self.age_holder))
		self.ai2_enc_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ai2,labels=tf.ones_like(ai2)/age_size))

		train_cls = tf.train.AdamOptimizer(0.0001).minimize(self.ai2_cls_loss,\
			var_list=M.get_all_vars('age_cls'))
		train_enc = tf.train.AdamOptimizer(0.0001).minimize(self.ai2_enc_loss,\
			var_list=M.get_all_vars('encoder'))

		with tf.control_dependencies([train_cls,train_enc]):
			self.train_ai2 = tf.no_op()

	def build_loss_A(self):
		tv_loss = tf.reduce_mean(tf.image.total_variation(self.A))
		l2_reg = tf.reduce_mean(tf.square(self.A))
		self.loss_A = tv_loss / (128*128) + l2_reg
		self.train_A = tf.train.AdamOptimizer(0.0001).minimize(self.loss_A,\
			var_list=M.get_all_vars('gen_att'))

	def update_ops(self):
		with tf.control_dependencies(M.get_update_ops()):
			self.update_bn = tf.no_op()

	def train(self, img, target, uniform, age_lab, id_lab, normalize=True):
		if normalize:
			img = np.float32(img) / 127.5 - 1.
			target = np.float32(target) / 127.5 - 1.
		else:
			img = np.float32(img)
			target = np.float32(target)

		feed_dict = {self.inp_holder:img, self.target_holder:target,\
		 self.uni_holder:uniform, self.age_holder: age_lab,\
		 self.id_holder:id_lab}

		fetches = [self.mc_loss, self.adv1_loss_g, self.adv1_loss_d, self.ip_loss,\
		self.adv2_loss_g, self.adv2_loss_d, self.age_cls_loss, self.age_generate_loss,\
		self.ai1_loss, self.ai2_cls_loss, self.ai2_enc_loss, self.loss_A,\
		self.train_mc, self.train_adv1, self.train_adv2, self.train_ip, self.train_ae,\
		self.train_ai1, self.train_ai2, self.train_A, self.update_bn,\
		self.generated]
		res = self.sess.run(fetches, feed_dict=feed_dict)
		return res[:12],res[-1]

	def save(self,modelname):
		print('Saving model to:',self.model_path+modelname)
		self.saver.save(self.sess, self.model_path+modelname)

	def eval(self, img, age_lab, normalize=True):
		if normalize:
			img = np.float32(img) / 127.5 - 1.
		else:
			img = np.float32(img)

		feed_dict = {self.inp_holder:img, self.age_holder: age_lab}
		res = self.sess.run([self.feat,self.generated],feed_dict=feed_dict)
		return res[0],res[1]

	def eval_Att(self, img, age_lab, normalize=True):
		if normalize:
			img = np.float32(img) / 127.5 - 1.
		else:
			img = np.float32(img)

		feed_dict = {self.inp_holder:img, self.age_holder: age_lab}
		res = self.sess.run([self.A,self.C],feed_dict=feed_dict)
		return res[0],res[1]

	@staticmethod
	def display_losses(losses):
		print('MC loss:\t%.4f'%losses[0])
		print('Adv1_G:\t%.4f\tAdv1_D:\t%.4f'%(losses[1],losses[2]))
		print('IP loss:\t%.4f'%losses[3])
		print('Adv2_G:\t%.4f\tAdv2_D:\t%.4f'%(losses[4],losses[5]))
		print('Age_class:\t%.4f\tAge_generate:\t%.4f'%(losses[6],losses[7]))
		print('Ai1 loss:\t%.4f'%losses[8])
		print('Ai2_class:\t%.4f\tAi2_encode:\t%.4f'%(losses[9],losses[10]))
		print('Att loss:\t%.4f'%losses[11])

	@staticmethod
	def denorm_img(inp):
		res = inp * 127.5 + 127.5
		res = np.uint8(res)
		return res

	@staticmethod
	def expand(src, target):
		target_shape = target.get_shape().as_list()
		src = tf.expand_dims(src,1)
		src = tf.expand_dims(src,1)
		src = tf.tile(src,[1,target_shape[1],target_shape[2],1])
		return src

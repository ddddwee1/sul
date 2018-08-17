import tensorflow as tf 
import model as M 
import network as N 
import numpy as np 

N.bn_training = True

class AIM_gen():
	def __init__(self,age_size,model_path = './aim_model_gen/'):
		self.model_path = model_path
		self.inp_holder = tf.placeholder(tf.float32,[None,128,128,3])
		self.age_holder = tf.placeholder(tf.float32,[None,1])
		self.age_holder2 = tf.placeholder(tf.float32,[None,1])

		# get_feature
		feat = N.feat_encoder(self.inp_holder)
		# get attention A and C
		age_expanded = self.expand(self.age_holder, feat)
		aged_feature = tf.concat([age_expanded, feat],-1)
		A, C = N.generator_att(aged_feature)
		# construct synthesized image
		generated = A * C + (1.-A) * self.inp_holder

		# get feature2
		feat2 = N.feat_encoder(generated)
		# get attention A2 and C2
		age_expanded2 = self.expand(self.age_holder2, feat2)
		aged_feature2 = tf.concat([age_expanded2, feat2],-1)
		A2, C2 = N.generator_att(aged_feature2)
		generated2 = A2*C2 + (1.-A2)*generated

		# retrieve tensor for adv2 and ae
		adv2, age_pred = N.discriminator(generated, age_size)
		adv2_real, age_pred_real = N.discriminator(self.inp_holder, age_size)

		adv2_2, age_pred2 = N.discriminator(generated2, age_size)

		# get gradient penalty

		gamma1 = tf.random_uniform([],0.0,1.0)
		interp1 = gamma1 * generated + (1. - gamma1) * self.inp_holder
		interp1_y, _ = N.discriminator(interp1, 7)
		grad_p1 = tf.gradients(interp1_y, interp1)[0]
		grad_p1 = tf.sqrt(tf.reduce_sum(tf.square(grad_p1),axis=[1,2,3]))
		grad_p1 = tf.reduce_mean(tf.square(grad_p1 - 1.) * 10.)

		gamma2 = tf.random_uniform([],0.0,1.0)
		interp2 = gamma2 * generated + (1. - gamma2) * self.inp_holder
		interp2_y, _ = N.discriminator(interp2, 7)
		grad_p2 = tf.gradients(interp2_y, interp2)[0]
		grad_p2 = tf.sqrt(tf.reduce_sum(tf.square(grad_p2),axis=[1,2,3]))
		grad_p2 = tf.reduce_mean(tf.square(grad_p2 - 1.) * 10.)


		# call loss builder functions
		self.mc_loss, self.train_mc = self.build_loss_mc(generated2, self.inp_holder)
		self.adv2_loss_d1, self.adv2_loss_g1, self.train_adv2_1 = self.build_loss_adv2(adv2, adv2_real, grad_p1)
		self.adv2_loss_d2, self.adv2_loss_g2, self.train_adv2_2 = self.build_loss_adv2(adv2_2,adv2_real, grad_p2)
		self.age_cls_loss_dis, self.train_ae_dis = self.build_loss_ae_dis(age_pred_real, self.age_holder2)
		self.age_cls_loss_gen, self.train_ae_gen = self.build_loss_ae_gen(age_pred, self.age_holder)
		self.age_cls_loss_gen2,self.train_ae_gen2= self.build_loss_ae_gen(age_pred2,self.age_holder2)
		self.loss_A, self.train_A = self.build_loss_A(A)
		self.loss_A2,self.train_A2=self.build_loss_A(A2)
		self.update_ops()
		self.accuracy = M.accuracy(age_pred_real, tf.argmax(self.age_holder2,-1))
		self.A1_l, self.A2_l = tf.reduce_mean(tf.square(A)), tf.reduce_mean(tf.square(A2))

		self.generated = generated
		self.A, self.C = A,C

		self.sess = tf.Session()
		M.loadSess(model_path,self.sess,init=True)
		M.loadSess('./aim_model/',self.sess,var_list=M.get_all_vars('encoder'))
		self.saver = tf.train.Saver()

	def build_loss_mc(self,generated,target):
		mc_loss = tf.reduce_mean(tf.abs(generated - target))
		train_mc = tf.train.AdamOptimizer(0.0000,beta1=0.5).minimize(mc_loss,var_list=M.get_all_vars('gen_att'))
		# train_mc = tf.no_op()
		with tf.control_dependencies([train_mc]):
			train_mc = tf.no_op()
		return mc_loss, train_mc

	def build_loss_adv2(self,adv2,adv2_real,grad_penalty):
		# adv2_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=adv2,labels=tf.zeros_like(adv2)))
		# adv2_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=adv2_real,labels=tf.ones_like(adv2_real)))
		# adv2_loss_d = 0.5 * (adv2_loss1 + adv2_loss2)

		# adv2_loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=adv2,labels=tf.ones_like(adv2)))

		adv2_loss_g = -tf.reduce_mean(adv2)
		adv2_loss_d = -tf.reduce_mean(adv2_real) + tf.reduce_mean(adv2)

		# adv2_loss_d += ddx 
		print('-----build loss finished, start optimizer')

		train_adv2_g = tf.train.AdamOptimizer(0.00005,beta1=0.5).minimize(adv2_loss_g,\
			var_list=M.get_all_vars('gen_att'))
		print('-----opt _g')
		with tf.variable_scope('adam_adv2_d',reuse=tf.AUTO_REUSE):
			train_adv2_d = tf.train.AdamOptimizer(0.00005,beta1=0.5).minimize(adv2_loss_d + grad_penalty,\
				var_list=M.get_all_vars('discriminator'))
		print('-----opt _d')

		with tf.control_dependencies([train_adv2_d,train_adv2_g]):
			train_adv2 = tf.no_op()

		return adv2_loss_d,adv2_loss_g,train_adv2

	def build_loss_ae_gen(self,age_pred, age_holder):
		# age_generate_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=age_pred,labels=age_holder))
		age_generate_loss = tf.reduce_mean(tf.abs(age_pred - age_holder))

		train_ae = tf.train.AdamOptimizer(0.0003,beta1=0.5).minimize(age_generate_loss,\
			var_list=M.get_all_vars('gen_att'))
		return age_generate_loss,train_ae

	def build_loss_ae_dis(self,age_pred, age_holder):
		# age_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=age_pred,labels=age_holder))
		age_loss = tf.reduce_mean(tf.abs(age_pred - age_holder))

		train_ae = tf.train.AdamOptimizer(0.0005,beta1=0.5).minimize(age_loss,\
			var_list=M.get_all_vars('discriminator'))
		return age_loss,train_ae

	def build_loss_A(self,A):
		tv_loss = tf.reduce_mean(tf.image.total_variation(A))
		l2_reg = tf.reduce_mean(tf.square(A))
		loss_A = tv_loss / (128.*128*10) + l2_reg * 1.0
		train_A = tf.train.AdamOptimizer(0.00006,beta1=0.5).minimize(loss_A,\
			var_list=M.get_all_vars('gen_att'))
		return loss_A,train_A

	def update_ops(self):
		with tf.control_dependencies(M.get_update_ops()):
			self.update_bn = tf.no_op()

	def train(self, img, age_lab, age_lab2, normalize=True):
		if normalize:
			img = np.float32(img) / 127.5 - 1.
		else:
			img = np.float32(img)

		feed_dict = {self.inp_holder:img, self.age_holder:age_lab,\
		 self.age_holder2: age_lab2}

		fetches = [self.mc_loss, self.adv2_loss_d1, self.adv2_loss_g1, self.adv2_loss_d2, self.adv2_loss_g2,\
		self.age_cls_loss_dis, self.age_cls_loss_gen, self.age_cls_loss_gen2, \
		self.accuracy, self.A1_l, self.A2_l,\
		self.train_adv2_1, self.train_adv2_2, self.train_ae_dis, self.train_ae_gen, self.train_ae_gen2,\
		self.update_bn,self.train_A, self.train_A2]
		res = self.sess.run(fetches, feed_dict=feed_dict)
		return res[:11]

	def save(self,modelname):
		print('Saving model to:',self.model_path+modelname)
		self.saver.save(self.sess, self.model_path+modelname)

	def eval(self, img, age_lab, normalize=True):
		if normalize:
			img = np.float32(img) / 127.5 - 1.
		else:
			img = np.float32(img)

		feed_dict = {self.inp_holder:img, self.age_holder: age_lab}
		res = self.sess.run(self.generated,feed_dict=feed_dict)
		resimg  = res + 1.
		resimg = resimg * 127.5
		resimg = np.uint8(resimg)
		print(resimg.shape)
		return resimg

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
		print('Adv_dis1:\t%.4f\tAdv_gen1:\t%.4f'%(losses[1],losses[2]))
		print('Adv_dis2:\t%.4f\tAdv_gen2:\t%.4f'%(losses[3],losses[4]))
		print('Age_dis:\t%.4f\tAge_acc:\t%.4f'%(losses[5],losses[8]))
		print('Age_gen1:\t%.4f\tAge_gen2:\t%.4f'%(losses[6],losses[7]))
		print('A1_length:\t%.4f\tA2_length:\t%.4f'%(losses[9],losses[10]))

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

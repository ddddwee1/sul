import tensorflow as tf 
import model2 as M 
import numpy as np 
import cv2

class GAN(M.Network):
	def __init__(self):
		self.model_path = './model/'
		super().__init__()

	def build_structure(self):
		self.input_holder = tf.placeholder(tf.float32,[None, 64])
		self.image_holder = tf.placeholder(tf.float32,[None, 64, 64, 3])
		
		self.recon = self.gen(self.input_holder)
		self.dis_g = self.dis(self.recon)
		self.dis_r = self.dis(self.image_holder)

	def gen(self, inp):
		with tf.variable_scope('gen'):
			mod = M.Model(inp)
			mod.fcLayer(4*4*256,activation=M.PARAM_LRELU)
			mod.reshape([-1,4,4,256])
			mod.deconvLayer(5,128,stride=2,activation=M.PARAM_LRELU, batch_norm=True) # 8
			mod.deconvLayer(5,64,stride=2,activation=M.PARAM_LRELU, batch_norm=True) # 16
			mod.deconvLayer(5,64,stride=2,activation=M.PARAM_LRELU, batch_norm=True) # 32
			mod.deconvLayer(5,32,stride=2,activation=M.PARAM_LRELU, batch_norm=True) # 64
			mod.convLayer(5,3,activation=M.PARAM_TANH)
		return mod.get_current_layer()

	def dis(self, inp):
		with tf.variable_scope('dis', reuse=tf.AUTO_REUSE):
			mod = M.Model(inp)
			mod.convLayer(5,32,stride=2,activation=M.PARAM_LRELU) # 32
			mod.convLayer(5,32,stride=2,activation=M.PARAM_LRELU, batch_norm=True) # 16
			mod.convLayer(5,64,stride=2,activation=M.PARAM_LRELU, batch_norm=True) # 8
			mod.convLayer(5,64,stride=2) # 4
		return mod.get_current_layer()

	def build_loss(self):
		self.loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dis_g, labels=tf.ones_like(self.dis_g)))
		self.train_g = tf.train.AdamOptimizer(0.0001).minimize(self.loss_g, var_list=M.get_trainable_vars('gen'))

		loss_d_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dis_g, labels=tf.zeros_like(self.dis_g)))
		loss_d_r = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dis_r, labels=tf.ones_like(self.dis_r)))
		self.loss_d = 0.5 * (loss_d_g + loss_d_r)
		self.train_d = tf.train.AdamOptimizer(0.0001).minimize(self.loss_d, var_list=M.get_trainable_vars('dis'))

		with tf.control_dependencies(M.get_update_ops()+[self.train_g, self.train_d]):
			self.train_op = tf.no_op()


	def train(self, z ,x, norm=False):
		x = np.float32(x)
		if norm:
			x = x /127.5 - 1
		lsg, lsd, rc, _ = self.sess.run([self.loss_g, self.loss_d, self.recon, self.train_op], feed_dict= {self.input_holder: z, self.image_holder:x})
		return lsg, lsd, rc

	def eval(self,x, norm=False):
		if norm:
			x = x /127.5 - 1
		rc = self.sess.run(self.recon, feed_dict= {self.input_holder: x})
		return rc

class reader(M.data_reader):
	def load_data(self):
		f = open('list.txt')
		for i in f:
			i = i.strip()
			img = cv2.imread(i,1)
			self.data.append([img, 0])
		print('data Loaded')

BSIZE = 64
net = GAN()
data = reader()

for i in range(1000*data.get_train_iter(BSIZE)):
	x_train, _ = data.get_next_batch(BSIZE)
	z = np.random.uniform(low=-1., high=1., size=[BSIZE, 64])
	lsg, lsd, rc = net.train(z, x_train, True)
	if i%10==0:
		print('Iter:%d\tLSG:%.4f\tLSD:%.4f'%(i,lsg,lsd))
	if i%1000==0:
		# save img
		rc = rc * 127.5 + 127.5 
		rc = np.uint8(rc)
		for j in range(len(rc)):
			cv2.imwrite('./gen/%d_%d.jpg'%(i,j), rc[j])
	if i%5000==4999:
		net.save('%d.ckpt'%(i+1))

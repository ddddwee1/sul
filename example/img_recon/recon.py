import tensorflow as tf 
import model2 as M 
import numpy as np 
import cv2
import network
import random

class recon_net(M.Network):
	def __init__(self):
		self.model_path = './model/'
		self.glob_step = 0
		self.apply_grad_period = 20
		super().__init__()

	def build_structure(self):
		self.image_holder = tf.placeholder(tf.float32,[None, None, None, 3])
		
		enc = network.encoder(self.image_holder)
		self.encoding = enc.get_current_layer()

		dec = network.decoder(self.encoding)
		self.reconstruct = dec.get_current_layer()

	def build_loss(self):
		self.loss = tf.reduce_mean(tf.abs(self.reconstruct - self.image_holder))
		trainer = M.Trainer(0.001, self.loss)
		self.accum_op = trainer.accumulate()
		self.train_op = trainer.train()

	def train(self, x, norm=True):
		x = np.float32(x)
		if norm:
			x = x /127.5 - 1
		self.glob_step += 1
		if self.glob_step%self.apply_grad_period == 0:
			ls, rc, _ = self.sess.run([self.loss, self.reconstruct, self.train_op], feed_dict= {self.image_holder:x})
		else:
			ls, rc, _ = self.sess.run([self.loss, self.reconstruct, self.accum_op], feed_dict= {self.image_holder:x})
		return ls, rc

	def eval(self,x, norm=True):
		x = np.float32(x)
		if norm:
			x = x /127.5 - 1
		rc = self.sess.run(self.reconstruct, feed_dict= {self.image_holder: x})
		return rc

class reader(M.list_reader):
	def load_data(self):
		f = open('list.txt')
		for i in f:
			i = i.strip()
			self.data.append([i, 0])
		random.shuffle(self.data)
		print('data Loaded')

	def process_img(self,i):
		img = cv2.imread(i,1)
		h,w,c = img.shape
		h1 = h-1 if h%2==1 else h
		w1 = w-1 if w%2==1 else w
		return img[:h1,:w1]

net = recon_net()
data = reader()

# for i in range(100*data.get_train_iter(1)):
for i in range(20000):
	x_train, _ = data.get_next_batch(1)
	ls, rc = net.train(x_train, True)
	if i%10==0:
		print('Iter:%d\tLS:%.4f'%(i,ls))
	if i%1000==0:
		# save img
		rc = rc * 127.5 + 127.5 
		rc = np.uint8(rc)
		for j in range(len(rc)):
			cv2.imwrite('./gen/%d_%d_gt.jpg'%(i,j), x_train[j])
			cv2.imwrite('./gen/%d_%d.jpg'%(i,j), rc[j])

		img = cv2.imread('1.png')
		rc = net.eval([img], True)
		rc = rc * 127.5 + 127.5 
		rc = np.uint8(rc)[0]
		cv2.imwrite('./gen/rc_%d.jpg'%i, rc)
	if i%5000==4999:
		net.save('%d.ckpt'%(i+1))

# img = cv2.imread('1.png')
# rc = net.eval([img], True)
# rc = rc * 127.5 + 127.5 
# rc = np.uint8(rc)[0]
# cv2.imwrite('./gen/rc.jpg', rc)

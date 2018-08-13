import network as N 
import numpy as np 
import model as M 
import tensorflow as tf 
import cv2

class recon():
	def __init__(self, step):
		with tf.variable_scope('Input_holders'):
			self.inpholder = tf.placeholder(tf.float32,[None,step,256,256,3])
			self.step = step
			self.targetholder = tf.placeholder(tf.float32,[None,step,256,256,3])

		with tf.variable_scope('Head'):
			inp_split = tf.unstack(self.inpholder,axis=1)
			features = []
			for i in range(len(inp_split)):
				features.append(N.conv_layers(inp_split[i],i!=0))
			features = tf.stack(features,axis=1)

		lstm_out = M.SimpleLSTM(4*4*32).apply(features)

		with tf.variable_scope('Tail'):
			feat_split = tf.unstack(lstm_out,axis=1)
			# I try the last frame for now
			feat = feat_split[-1]
			A,C = N.deconv_layers(feat)

			self.recon = A * C + (1. - A) * tf.reduce_mean(self.inpholder,axis=1)

		self.A = A
		self.build_loss()

		self.sess = tf.Session()
		M.loadSess('./model/',self.sess,init=True)
		self.saver = tf.train.Saver()

	def build_loss(self):
		with tf.variable_scope('LS_optim'):
			self.ls = tf.reduce_mean(tf.square(self.recon - self.targetholder[:,-1]))
			self.A_length = tf.reduce_mean(tf.square(self.A))
			train_step = tf.train.AdamOptimizer(0.0001).minimize(self.ls)
			with tf.control_dependencies(M.get_update_ops()+[train_step]):
				self.train_step = tf.no_op()

	def train(self, inp, target, normalize=True):
		inp = np.float32(inp)
		target = np.float32(target)
		if normalize:
			inp = inp / 127.5 - 1.
			target = target / 127.5 - 1.

		ls, A_length, _ = self.sess.run([self.ls, self.A_length, self.train_step],feed_dict={self.inpholder:inp, self.targetholder:target})
		return ls,A_length

	def save(self,name):
		self.saver.save(self.sess,'./model/'+name)

	def eval(self,inp,normalize=True):
		inp = np.float32(inp)
		if normalize:
			inp = inp / 127. - 1.
		res = self.sess.run(self.recon,feed_dict={self.inpholder:inp})
		res = res + 1.
		res = res * 127.5
		res = np.uint8(res)[0]
		return res

	def eval_A(self,inp,normalize=True):
		inp = np.float32(inp)
		if normalize:
			inp = inp / 127. - 1.
		res = self.sess.run(self.A,feed_dict={self.inpholder:inp})
		res = res * 255
		res = np.uint8(res)[0]
		return res

import data_reader

BSIZE = 16
STEP = 5
data_reader = data_reader.data_reader()
recon_net = recon(STEP)

MAXITER = 100000
eta = M.ETA(MAXITER)
for i in range(MAXITER+1):
	src, tgt = data_reader.get_batch(BSIZE, STEP)
	ls,A_length = recon_net.train(src, tgt)
	print('Iter:%4d\tLoss:%.5f\tA_length:%.5f\tETA:%s'%(i,ls,A_length,eta.get_ETA(i)))
	if i%1000==0 and i>0:
		recon_net.save('%d.ckpt'%i)
	if i%10==0:
		src,_ = data_reader.get_batch(1,STEP)
		img = recon_net.eval(src)
		img2 = recon_net.eval_A(src)
		cv2.imwrite('./res/%d.jpg'%i,img)
		cv2.imwrite('./res/%d_A.png'%i,img2)

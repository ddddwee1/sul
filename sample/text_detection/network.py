import tensorflow as tf 
import model as M 
import numpy as np 
import Res50

class text_net():
	def __init__(self):
		self.image_holder = tf.placeholder(tf.float32,[None, 512, 512, 3])
		self.mask_holder = tf.placeholder(tf.float32,[None, 128, 128])
		self.geo_holder = tf.placeholder(tf.float32,[None, 128, 128, 4])
		self.rot_holder = tf.placeholder(tf.float32,[None, 128, 128, 1])
		self.quad_holder = tf.placeholder(tf.float32, [None, 128, 128, 8])

		f1,f2,f3,f4 = Res50.build_model(self.image_holder)
		output = self.feature_merging_branch(f1,f2,f3,f4)

		self.mask_output = output[:,:,:,0:1]
		self.geo_output = output[:,:,:,1:5]
		self.rot_output = output[:,:,:,5:6]
		self.quad_output = output[:,:,:,6:14]

		self.losses, self.train_step = self.get_train_step()

		self.sess = tf.Session()
		self.saver = tf.train.Saver()
		M.loadSess('./model/', sess=self.sess, init=True)

	def feature_merging_branch(self, f1,f2,f3,f4):
		# merge features and give output of shape [BSIZE, h, w, 14] 
		pass 

	def get_train_step():
		# build loss functions and get train step
		mask_loss = self.dice_loss(self.mask_output, self.mask_holder)
		geo_loss = self.AABB_loss(self.geo_output, self.geo_holder, self.mask_holder)
		rot_loss = self.theta_loss(self.rot_output, self.rot_holder, self.mask_holder)
		quad_loss = self.quad_regression_loss(self.quad_output, self.quad_holder, self.mask_holder)
		
		losses = [mask_loss, geo_loss, rot_loss, quad_loss]
		weights = [1,1,1,1] # modify in future

		loss_total = sum([w*l for w,l in zip(weights, losses)])
		with tf.control_dependencies( M.get_update_ops()):
			self.train_op = tf.train.AdamOptimizer(0.0001, 0.9).minimize(loss_total)

	def dice_loss(self, output, label):
		# code for dice loss
		pass 

	def AABB_loss(self, output, label, mask):
		# code for AABB loss
		pass 

	def theta_loss(self, output, label, mask):
		# code for theta loss
		loss = 1. - tf.cos(output - label)
		loss = loss*mask
		return loss 

	def quad_regression_loss(self, output, label, mask):
		# code for quad regression loss
		pass

	def train(self, image, geo_label, mask_label, theta_label, quad_label):
		# function wrap-up for training process
		pass 

	def val(self, image, geo_label, mask_label, theta_label, quad_label):
		pass

	def get_output(self, image):
		# get output of network
		pass 

	def save(self, name):
		self.saver.save(self.sess, './model/'+name)

	def print_losses(self, losses):
		# print losses
		pass 

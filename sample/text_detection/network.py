import tensorflow as tf 
import model as M 
import numpy as np 
import Res50

class text_net():
	def __init__(self):
		self.image_holder = tf.placeholder(tf.float32,[None, 512, 512, 3])
		self.mask_holder = tf.placeholder(tf.float32,[None, 128, 128, 1])
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
		with tf.variable_scope('feature_merging'):
			mod = M.Model(f1)
			mod.resize([32,32])
			mod.concat_to_current(f2)
			mod.convLayer(1, 128, activation=M.PARAM_RELU)
			mod.convLayer(3, 128, activation=M.PARAM_RELU)
			mod.resize([64,64])
			mod.concat_to_current(f3)
			mod.convLayer(1, 64, activation=M.PARAM_RELU)
			mod.convLayer(3, 64, activation=M.PARAM_RELU)
			mod.resize([128,128])
			mod.concat_to_current(f4)
			mod.convLayer(1, 32, activation=M.PARAM_RELU)
			mod.convLayer(3, 32, activation=M.PARAM_RELU)
			mod.convLayer(3, 32, activation=M.PARAM_RELU)
			mod.convLayer(1, 14)
		return mod.get_current_layer()

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
			train_op = tf.train.AdamOptimizer(0.0001, 0.9).minimize(loss_total)
		return losses, train_op

	def dice_loss(self, output, label):
		# code for dice loss
		inter = tf.reduce_sum(output * label, [1,2,3])
		union = tf.reduce_sum(output, [1,2,3]) + tf.reduce_sum(label, [1,2,3]) - inter 
		loss = 1. - inter / union 
		loss = tf.reduce_mean(loss)
		return loss 

	def AABB_loss(self, output, label, mask):
		# code for AABB loss
		a_out = (output[:,:,:,0] + output[:,:,:,2]) * (output[:,:,:,1] + output[:,:,:,3])
		a_gt = (label[:,:,:,0] + label[:,:,:,2]) * (label[:,:,:,1] + label[:,:,:,3])
		out_gt_stack = tf.stack([output, label], axis=0)
		inter = tf.reduce_min(out_gt_stack, axis=0)
		a_inter = (inter[:,:,:,0] + inter[:,:,:,2]) * (inter[:,:,:,1] + inter[:,:,:,3])
		a_union = a_out + a_gt - a_inter
		loss = -tf.log((a_inter + 1.0)/(a_union + 1.0))
		loss = tf.reduce_mean(loss)
		return loss

	def theta_loss(self, output, label, mask):
		# code for theta loss
		loss = 1. - tf.cos(output - label)
		loss = tf.reduce_mean(loss*mask)
		return loss 

	def quad_regression_loss(self, output, label, mask):
		# code for quad regression loss
		pass

	def train(self, image, geo_label, mask_label, rot_label, quad_label):
		# function wrap-up for training process
		feed_dict = {self.image_holder: image, self.geo_holder: geo_label, self.mask_holder: mask_label, self.rot_holder: rot_label, self.quad_holder:quad_label}
		output_tensors = self.losses + [self.train_step]
		output = self.sess.run(output_tensors, feed_dict=feed_dict)
		output = output[:-1]
		return output

	def val(self, image, geo_label, mask_label, rot_label, quad_label):
		feed_dict = {self.image_holder: image, self.geo_holder: geo_label, self.mask_holder: mask_label, self.rot_holder: rot_label, self.quad_holder:quad_label}
		output_tensors = self.losses
		output = self.sess.run(output_tensors, feed_dict=feed_dict)
		return output

	def get_output(self, image):
		# get output of network
		feed_dict = {self.image_holder: image}
		output_tensors = [self.mask_output,self.geo_output,self.rot_output,self.quad_output]
		output = self.sess.run(output_tensors, feed_dict=feed_dict)
		return output

	def save(self, name):
		self.saver.save(self.sess, './model/'+name)

	def print_losses(self, losses):
		# print losses [mask_loss, geo_loss, rot_loss, quad_loss]
		mask_loss, geo_loss, rot_loss, quad_loss = losses 
		print('Mask:%.4f\tGeo:%.4f\tRot:%.4f\tQuad:%.4f'%(mask_loss, geo_loss, rot_loss, quad_loss))

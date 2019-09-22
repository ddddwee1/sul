import layers3 as L
import model3 as M 
import random
import tensorflow as tf 
import numpy as np 

class ResBlock1D(M.Model):
	def initialize(self, outchn=1024, dilation=1, k=3):
		self.bn = M.BatchNorm()
		self.activ = L.activation(M.PARAM_LRELU)
		self.c1 = M.ConvLayer1D(k, outchn, dilation_rate=dilation, activation=M.PARAM_LRELU, batch_norm=True)
		self.c2 = M.ConvLayer1D(3, outchn)

	def forward(self, x):
		short = x

		# residual branch
		branch = self.bn(x)
		branch = self.activ(branch)
		branch = self.c1(branch)
		branch = self.c2(branch)

		# slicing & shortcut
		# branch_shape = branch.get_shape().as_list()[1]
		# short_shape = short.get_shape().as_list()[1]
		# start = (short_shape - branch_shape) // 2
		# short = short[:,start:start+branch_shape, :]
		res = short + branch
		res = tf.nn.dropout(res, 0.25)
		return res

class Refine2dNet(M.Model):
	def initialize(self):
		self.c1 = M.ConvLayer1D(5,1024, activation=M.PARAM_LRELU, batch_norm=True)
		self.r1 = ResBlock1D()
		self.r2 = ResBlock1D()
		self.r3 = ResBlock1D()
		self.c4 = M.ConvLayer1D(1, 17*2)

	def forward(self, x, drop=True):
		x = self.c1(x)
		x = self.r1(x)

		x = self.r2(x)
		x = self.r3(x)
		x = self.c4(x)
		return x 

class Discriminator2D(M.Model):
	def initialize(self):
		self.c1 = M.ConvLayer1D(5, 1024, stride=2, activation=M.PARAM_LRELU, batch_norm=True)
		self.c2 = M.ConvLayer1D(5, 256, stride=2, activation=M.PARAM_LRELU, batch_norm=True)
		self.c3 = M.ConvLayer1D(5, 256, stride=2, activation=M.PARAM_LRELU, batch_norm=True)
		self.c4 = M.ConvLayer1D(1, 1)

	def forward(self, x):
		return self.c4(self.c3(self.c2(self.c1(x))))

class DepthEstimator(M.Model):
	def initialize(self):
		self.c1 = M.ConvLayer1D(5, 1024, activation=M.PARAM_LRELU, batch_norm=True)
		self.r1 = ResBlock1D()
		self.r2 = ResBlock1D()
		self.r3 = ResBlock1D()
		self.c4 = M.ConvLayer1D(1, 17)

	def forward(self, x, drop=True):
		x = self.c1(x)
		x = self.r1(x)
		x = self.r2(x)
		x = self.r3(x)
		x = self.c4(x)
		return x 

class Discriminator3D(M.Model):
	def initialize(self):
		self.c1 = M.ConvLayer1D(5, 1024, stride=2, activation=M.PARAM_LRELU, batch_norm=True)
		self.c2 = M.ConvLayer1D(5, 256, stride=2, activation=M.PARAM_LRELU, batch_norm=True)
		self.c3 = M.ConvLayer1D(5, 256, stride=2, activation=M.PARAM_LRELU, batch_norm=True)
		self.c4 = M.ConvLayer1D(1, 1)

	def forward(self, x):
		return self.c4(self.c3(self.c2(self.c1(x))))

def disLoss(real, fake):
	loss_d_real = tf.reduce_mean(tf.square( real - tf.ones_like(real) ))
	loss_d_fake = tf.reduce_mean(tf.square( fake - tf.zeros_like(fake)))
	loss_d = (loss_d_real + loss_d_fake) * 0.5

	loss_g = tf.reduce_mean(tf.square( fake - tf.ones_like(fake) ))
	return loss_d, loss_g

def MSELoss(logit, label):
	return tf.reduce_mean(tf.square(logit - label))

def reproj(pts_2d, depth, rot=None):
	if rot is None:
		rot = random.random() * 2 - 1
	rot = rot * np.pi

	pts_2d_x = pts_2d[:,:,0::2]
	pts_2d_y = pts_2d[:,:,1::2]
	pts_2d_x_1 = tf.cos(rot) * pts_2d_x - tf.sin(rot) * depth
	depth_1 = tf.sin(rot) * pts_2d_x + tf.cos(rot) * depth

	pts_2d = tf.stack([pts_2d_x_1, pts_2d_y], axis=-1)
	pts_2d = tf.reshape(pts_2d, [depth.shape[0], -1 , 17*2])
	return pts_2d, depth_1

def concat_3d(pts_2d, depth):
	pts_2d_x = pts_2d[:,:,0::2]
	pts_2d_y = pts_2d[:,:,1::2]

	pts_3d = tf.stack([pts_2d_x, pts_2d_y, depth], axis=-1)
	pts_3d = tf.reshape(pts_3d, [pts_3d.shape[0], -1, 17*3])
	return pts_3d

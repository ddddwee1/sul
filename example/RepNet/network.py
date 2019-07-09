import numpy as np 
import tensorflow as tf 
import model3 as M 

class DenseBlock(M.Model):
	def initialize(self):
		self.d1 = Dense(512, activation=M.PARAM_RELU)
		self.d2 = Dense(512, activation=M.PARAM_RELU)
	def forward(self, x):
		orign = x 
		x = self.d1(x)
		x = self.d2(x)
		x = orign + x 
		return x 

class GenNet(M.Model):
	def initialize(self):
		self.d1 = Dense(512, activation=M.PARAM_RELU)
		self.d2 = DenseBlock()
		self.d3 = DenseBlock()
		self.d4 = Dense(17*3)
	def forward(self, x):
		x = self.d1(x)
		x = self.d2(x)
		x = self.d3(x)
		x = self.d4(x)
		return x 

class CritNet(M.Model):
	def initialize(self, C):
		# Chain matrix
		self.C = C  # 17 x N
		self.bone_num = C.shape[-1]
		self.c1 = Dense(512, activation=M.PARAM_RELU)
		self.c2 = DenseBlock()

		self.d1 = Dense(512, activation=M.PARAM_RELU)
		self.d2 = DenseBlock()
		self.d3 = DenseBlock()
		self.d4 = Dense(1)
	def forward(self, x):
		# The Kinematic branch
		c = x 
		c = tf.reshape(c, [-1, 17, 3])
		c = tf.matmul(c, self.C, transpose_a=True)
		psi = tf.matmul(c, c, transpose_a=True)
		psi = tf.reshape(psi, [-1, self.bone_num*self.bone_num])
		psi = self.c1(psi)
		psi = self.c2(psi)

		# The normal branch
		x = self.d1(x)
		x = self.d2(x)

		# Final branch
		x = tf.concat([psi, x], axis=-1)
		x = self.d3(x)
		x = self.d4(x)
		return x 

class CamNet(M.Model):
	def initialize(self):
		self.d1 = Dense(512, activation=M.PARAM_RELU)
		self.d2 = DenseBlock()
		self.d3 = DenseBlock()
		self.d4 = Dense(6)
	def forward(self, x):
		x = self.d1(x)
		x = self.d2(x)
		x = self.d3(x)
		x = self.d4(x)
		return x 

class Reprojection(M.Model):
	def forward(self, x, K):
		K = tf.reshape(K, [3, 2])
		x = tf.reshape(x, [-1, 17, 3])
		x = tf.matmul(x, K)
		return x 

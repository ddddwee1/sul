import model3 as M 
import numpy as np 
import tensorflow as tf 


class ConvBlock(M.Model):
	def initialize(self, chn, outchn, dilation):
		self.c_filt = M.ConvLayer(2, chn, pad='VALID', dilation_rate=dilation, activation=M.PARAM_TANH)
		self.c_gate = M.ConvLayer(2, chn, pad='VALID', dilation_rate=dilation, activation=M.PARAM_SIGMOID)
		self.c_skip = M.ConvLayer(1, chn, pad='VALID')
		self.c_transform = M.ConvLayer(1, outchn, pad='VALID')

	def forward(self, x):
		filt = self.c_filt(x)
		gate = self.c_gate(x)
		res = filt * gate 

		# res skip 
		skip = self.c_skip(res)

		# output 
		transform = self.c_transform(res)
		outlength = transform.shape[1]
		out = transform + x[:,-outlength:]
		return skip, out 

class WaveNet(M.Model):
	def initialize(self):
		self.b1 = ConvBlock(32, 32, 1)
		self.b2 = ConvBlock(32, 32, 2)
		self.b3 = ConvBlock(32, 32, 4)
		self.b4 = ConvBlock(32, 32, 8)

		self.c1 = M.ConvLayer(1, 128, activation=M.PARAM_RELU)
		self.c2 = M.ConvLayer(1, 128)

	def forward(self, x):
		k1,o1 = self.b1(x)
		k2,o2 = self.b2(x)
		k3,o3 = self.b3(x)
		k4,o4 = self.b4(x)

		k = [k1, k2, k3, k4]
		outlen = k4.shape[1]
		k = [i[:,-outlen] for i in k]
		k = tf.concat(k, axis=2)

		k = self.c2(self.c1(k))
		return k 
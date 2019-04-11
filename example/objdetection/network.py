import tensorflow as tf 
import model3 as M 
import numpy as np 
import config 

class ResUnit(M.Model):
	def initialize(self, out, stride, shortcut=False):
		self.shortcut = shortcut
		self.c1 = M.ConvLayer(1, out//4, usebias=False, activation=M.PARAM_RELU, batch_norm=True)
		self.c2 = M.ConvLayer(3, out//4, usebias=False, activation=M.PARAM_RELU, pad='SAME_LEFT', stride=stride, batch_norm=True)
		self.c3 = M.ConvLayer(1, out, usebias=False, batch_norm=True)
		if shortcut:
			self.sc = M.ConvLayer(1, out, usebias=False, stride=stride, batch_norm=True)

	def forward(self, x):
		branch = self.c1(x)
		branch = self.c2(branch)
		branch = self.c3(branch)
		if self.shortcut:
			sc = self.sc(x)
		else:
			sc = x 
		res = branch + sc
		res = tf.nn.relu(res)
		return res 

class ResBlock(M.Model):
	def initialize(self, out, stride, num_units):
		self.units = []
		for i in range(num_units):
			self.units.append(ResUnit(out, stride if i==0 else 1, True if i==0 else False))
	def forward(self, x):
		for unit in self.units:
			x = unit(x)
		return x 

class ResNet(M.Model):
	def initialize(self):
		self.mean = tf.constant(np.float32(config.imagenet_mean))
		self.std = tf.constant(np.float32(config.imagenet_std))
		self.c1 = M.ConvLayer(7, 64, pad='SAME_LEFT', stride=2, activation=M.PARAM_RELU, usebias=False, batch_norm=True)
		self.mp1 = M.MaxPool(3,2, pad='SAME_LEFT')
		self.block1 = ResBlock(256, 1, 3)
		self.block2 = ResBlock(512, 2, 4)
		self.block3 = ResBlock(1024, 2, 6)
		self.block4 = ResBlock(2048, 2, 3)

	def forward(self, x):
		# pre-process
		x = x / 255.
		x = (x - self.mean) / self.std
		# end: pre-process
		x = self.c1(x)
		x = self.mp1(x)
		x = self.block1(x)
		# f4 = x
		x = self.block2(x)
		f3 = x
		x = self.block3(x)
		f2 = x
		x = self.block4(x)
		f1 = x
		return [f1,f2,f3]

class FeaturePyramid(M.Model):
	def initialize(self):
		self.c1 = M.ConvLayer(1, 256, activation=M.PARAM_RELU)
		self.c2_in = M.ConvLayer(1, 256, activation=M.PARAM_RELU)
		self.c2_aft = M.ConvLayer(1, 128, activation=M.PARAM_RELU)
		self.c3_in = M.ConvLayer(1, 128, activation=M.PARAM_RELU)
		self.c3_aft = M.ConvLayer(1, 128, activation=M.PARAM_RELU)
		self.upsample = M.BilinearUpSample(2)

	def forward(self, f1,f2,f3):
		res = self.c1(f1)
		res = self.upsample(res)
		f2 = self.c2_in(f2)
		res = res + f2 
		res = self.c2_aft(res)
		res = self.upsample(res)
		f3 = self.c3_in(f3)
		res = res + f3 
		res = self.c3_aft(res)
		return res 

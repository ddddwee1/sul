import tensorflow as tf 
import numpy as np 
import model3 as M 

class VGG19(M.Model):
	def initialize(self):
		self.c11 = M.ConvLayer(3, 64, activation=M.PARAM_RELU, batch_norm=True, usebias=False, pad='SAME_LEFT')
		self.c12 = M.ConvLayer(3, 64, activation=M.PARAM_RELU, batch_norm=True, usebias=False, pad='SAME_LEFT')
		self.m1 = M.MaxPool(2, 2)

		self.c21 = M.ConvLayer(3, 128, activation=M.PARAM_RELU, batch_norm=True, usebias=False, pad='SAME_LEFT')
		self.c22 = M.ConvLayer(3, 128, activation=M.PARAM_RELU, batch_norm=True, usebias=False, pad='SAME_LEFT')
		self.m2 = M.MaxPool(2, 2)

		self.c31 = M.ConvLayer(3, 256, activation=M.PARAM_RELU, batch_norm=True, usebias=False, pad='SAME_LEFT')
		self.c32 = M.ConvLayer(3, 256, activation=M.PARAM_RELU, batch_norm=True, usebias=False, pad='SAME_LEFT')
		self.c33 = M.ConvLayer(3, 256, activation=M.PARAM_RELU, batch_norm=True, usebias=False, pad='SAME_LEFT')
		self.c34 = M.ConvLayer(3, 256, activation=M.PARAM_RELU, batch_norm=True, usebias=False, pad='SAME_LEFT')
		self.m3 = M.MaxPool(2, 2)

		self.c41 = M.ConvLayer(3, 512, activation=M.PARAM_RELU, batch_norm=True, usebias=False, pad='SAME_LEFT')
		self.c42 = M.ConvLayer(3, 512, activation=M.PARAM_RELU, batch_norm=True, usebias=False, pad='SAME_LEFT')
		self.c43 = M.ConvLayer(3, 512, activation=M.PARAM_RELU, batch_norm=True, usebias=False, pad='SAME_LEFT')
		self.c44 = M.ConvLayer(3, 512, activation=M.PARAM_RELU, batch_norm=True, usebias=False, pad='SAME_LEFT')
		self.m4 = M.MaxPool(2, 2)

		self.fc1 = M.Dense(2048, batch_norm=True, usebias=False)
		self.fc2 = M.Dense(512)


	def forward(self, x):
		x = self.c11(x)
		x = self.c12(x)
		x = self.m1(x)

		x = self.c21(x)
		x = self.c22(x)
		x = self.m2(x)

		x = self.c31(x)
		x = self.c32(x)
		x = self.c33(x)
		x = self.c34(x)
		x = self.m3(x)

		x = self.c41(x)
		x = self.c42(x)
		x = self.c43(x)
		x = self.c44(x)
		x = self.m4(x)

		x = M.flatten(x)
		x = self.fc1(x)
		x = self.fc2(x)
		return x 

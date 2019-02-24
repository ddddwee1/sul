import layers2 as L 
import modeleag as M 
import tensorflow as tf

class network(M.Model):
	def initialize(self):
		self.c1 = M.ConvLayer(7, 32, stride=2, activation=M.PARAM_RELU)
		self.c2 = M.ConvLayer(5, 64, stride=1, activation=M.PARAM_RELU)
		self.c3 = M.ConvLayer(5, 128, stride=2, activation=M.PARAM_RELU)
		self.c33 = M.ConvLayer(5, 128, stride=1, activation=M.PARAM_RELU)
		self.c4 = M.ConvLayer(5, 128, stride=2, activation=M.PARAM_RELU)
		self.c5 = M.ConvLayer(5, 128, stride=1, activation=M.PARAM_RELU)
		self.c6 = M.ConvLayer(3, 256, stride=2, activation=M.PARAM_RELU)
		self.c7 = M.ConvLayer(3, 512, stride=1, dilation_rate=2, activation=M.PARAM_RELU)
		self.c8 = M.ConvLayer(3, 512, stride=1, dilation_rate=2, activation=M.PARAM_RELU)
		self.c9 = M.ConvLayer(1, 5, stride=1)

	def forward(self, x):
		x = self.c1(x)
		x = self.c2(x)
		x = self.c3(x)
		x = self.c33(x)
		x = self.c4(x)
		x = self.c5(x)
		x = self.c6(x)
		x = self.c7(x)
		x = self.c8(x)
		x = self.c9(x)
		return x 

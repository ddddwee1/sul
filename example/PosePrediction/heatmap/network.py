import numpy as np 
import model3 as M 
import tensorflow as tf 

EMB_DIM=512

class Encoder(M.Model):
	def initialize(self):
		self.c1 = M.ConvLayer(5, 64, stride=2, activation=M.PARAM_LRELU)
		self.c2 = M.ConvLayer(5, 128, stride=2, activation=M.PARAM_LRELU)
		self.c3 = M.ConvLayer(5, 128, stride=2, activation=M.PARAM_LRELU)
		self.c4 = M.ConvLayer(5, 128, stride=2, activation=M.PARAM_LRELU)
		self.c5 = M.Dense(EMB_DIM, activation=M.PARAM_TANH)
	def forward(self, x):
		x = self.c1(x)
		x = self.c2(x)
		x = self.c3(x)
		x = self.c4(x)
		x = M.flatten(x)
		x = self.c5(x)
		return x 

class Decoder(M.Model):
	def initialize(self, num_kpt):
		self.fc = M.Dense(8*8*128, activation=M.PARAM_LRELU)
		self.c0 = M.DeconvLayer(5, 128, stride=2, activation=M.PARAM_LRELU)
		self.c1 = M.DeconvLayer(5, 128, stride=2, activation=M.PARAM_LRELU)
		self.c2 = M.DeconvLayer(5, 64, stride=2, activation=M.PARAM_LRELU)
		self.c3 = M.ConvLayer(5, num_kpt)
	def forward(self, x):
		x = self.fc(x)
		x = tf.reshape(x, [-1,8,8,128])
		x = self.c0(x)
		x = self.c1(x)
		x = self.c2(x)
		x = self.c3(x)
		return x 

class TempModule(M.Model):
	def initialize(self):
		self.lstm = M.LSTM(EMB_DIM)
	def forward(self, x):
		out = self.lstm(x)
		return out 

class PosePred(M.Model):
	def initialize(self, num_kpt):
		self.enc = Encoder()
		self.temp = TempModule()
		self.dec = Decoder(num_kpt)
	def forward(self, x, extend, prednum):
		x = [self.enc(_) for _ in x]
		x = x + [None]*extend
		x = self.temp(x)
		x = x[-prednum:]
		x = [self.dec(_) for _ in x]
		return x 

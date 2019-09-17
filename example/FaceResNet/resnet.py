import tensorflow as tf 
import numpy as np 
import model3 as M 

class ResBlock_v1(M.Model):
	def initialize(self, outchn, stride, bottle_neck=False):
		self.stride = stride
		self.outchn = outchn
		self.bn1 = M.BatchNorm()
		self.c1 = M.ConvLayer(3, outchn, activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
		self.c2 = M.ConvLayer(3, outchn, stride=stride, usebias=False, batch_norm=True)

		# se module 
		# self.gp = M.GlobalAvgPool()
		# self.c3 = M.ConvLayer(1, outchn//16, activation=M.PARAM_LRELU)
		# self.c4 = M.ConvLayer(1, outchn, activation=M.PARAM_SIGMOID)

		# shortcut 
		self.sc = M.ConvLayer(1, outchn, stride=stride, usebias=False, batch_norm=True)

	def build(self, input_shape):
		self.inchn = input_shape[-1]

	def forward(self, x):
		res = self.bn1(x)
		res = self.c1(res)
		res = self.c2(res)
		# se
		# se = self.gp(res)
		# se = self.c3(se)
		# se = self.c4(se)
		# res = res * se 
		# shortcut 
		if self.inchn==self.outchn and self.stride==1:
			sc = x 
		else:
			sc = self.sc(x)
		res = res + sc 
		return res 

class HeadBlock(M.Model):
	def initialize(self, outchn, stride=1):
		self.c1 = M.ConvLayer(3, outchn, stride, usebias=False, activation=M.PARAM_LRELU, batch_norm=True)

	def forward(self, x):
		x = x - 127.5
		x = x * 0.0078125
		x = self.c1(x)
		return x

class Tail(M.Model):
	def initialize(self, classnum):
		self.fc = M.Dense(classnum)
	def forward(self, x):
		x = self.fc(x)
		return x 

class ResNet(M.Model):
	def initialize(self, channel_list, blocknum_list, embedding_size, embedding_bn=True):
		self.head = HeadBlock(channel_list[0])
		self.body = []
		for num, chn in zip(blocknum_list, channel_list[1:]):
			for i in range(num):
				self.body.append(ResBlock_v1(chn, 2 if i==0 else 1))

		self.emb_bn = M.BatchNorm()
		self.embedding = M.Dense(embedding_size, batch_norm=embedding_bn)

	def forward(self, x):
		x = self.head(x)
		for block in self.body:
			x = block(x)
		x = M.flatten(x)
		x = self.emb_bn(x)
		x = tf.nn.dropout(x,0.4)
		x = self.embedding(x)
		return x 

def Res100():
	return ResNet([64,64,128,256,512],[3,13,30,3],512)

def Res50():
	return ResNet([64,64,128,256,512],[3,4,14,3],512)
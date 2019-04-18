import tensorflow as tf 
import numpy as np 
import model3 as M 

class ResBlock_v1(M.Model):
	def initialize(self, outchn, stride, ratio, input_ratio=None, bottle_neck=False):
		self.stride = stride
		self.outchn = outchn
		self.bn1 = M.BatchNorm()
		self.c1 = M.OctConv(3, outchn, ratio, input_ratio, activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
		self.c2 = M.OctConv(3, outchn, ratio, input_ratio, stride=stride, usebias=False, batch_norm=True)
		# shortcut 
		self.sc = M.OctConv(1, outchn, ratio, input_ratio, stride=stride, usebias=False, batch_norm=True)

	def build(self, input_shape):
		self.inchn = input_shape[-1]

	def forward(self, x):
		res = self.bn1(x)
		res = self.c1(res)
		res = self.c2(res)
		# shortcut 
		if self.inchn==self.outchn and self.stride==1:
			sc = x 
		else:
			sc = self.sc(x)
		res = res + sc 
		return res 

class ResBlock_normal(M.Model):
	def initialize(self, outchn, stride, bottle_neck=False):
		self.stride = stride
		self.outchn = outchn
		self.bn1 = M.BatchNorm()
		self.c1 = M.ConvLayer(3, outchn, activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
		self.c2 = M.ConvLayer(3, outchn, stride=stride, usebias=False, batch_norm=True)
		# shortcut 
		self.sc = M.ConvLayer(1, outchn, stride=stride, usebias=False, batch_norm=True)

	def build(self, input_shape):
		self.inchn = input_shape[-1]

	def forward(self, x):
		res = self.bn1(x)
		res = self.c1(res)
		res = self.c2(res)
		# shortcut 
		if self.inchn==self.outchn and self.stride==1:
			sc = x 
		else:
			sc = self.sc(x)
		res = res + sc 
		return res 

class HeadBlock(M.Model):
	def initialize(self, outchn, size=3, stride=1, maxpool=True):
		self.c1 = M.ConvLayer(size, outchn, stride, usebias=False, activation=M.PARAM_LRELU, batch_norm=True)
		self.maxpool = maxpool
		if maxpool:
			self.mxpool = M.MaxPool(3,2)

	def forward(self, x):
		x = x - 127.5
		x = x * 0.0078125
		x = self.c1(x)
		if self.maxpool:
			x = self.mxpool(x)
		return x

class ResNet(M.Model):
	def initialize(self, channel_list, blocknum_list, embedding_size, oct_ratio, embedding_bn=True):
		self.head = HeadBlock(channel_list[0], size=3, stride=1, maxpool=False)
		self.pre_oct = M.OctMerge()
		if oct_ratio!=0.5:
			self.c0 = M.OctConv(1, channel_list[0], oct_ratio, 0.5)
		self.body = []
		for i_blk, (num, chn) in enumerate(zip(blocknum_list, channel_list[1:])):
			for i in range(num):
				if i_blk==3:
					if i==0:
						self.body.append(M.OctSplit(oct_ratio))
					blk = ResBlock_normal(chn, 1)
				else:
					blk = ResBlock_v1(chn, 2 if (i==0) else 1, oct_ratio)
				self.body.append(blk)

		self.emb_bn = M.BatchNorm()
		self.embedding = M.Dense(embedding_size, batch_norm=embedding_bn)

		self.oct_ratio = oct_ratio

	def forward(self, x):
		x = self.head(x)
		x = self.pre_oct(x)
		if self.oct_ratio!=0.5:
			x = self.c0(x)
		for block in self.body:
			x = block(x)
		print(x.shape)
		x = M.flatten(x)
		x = self.emb_bn(x)
		if tf.keras.backend.learning_phase():
			x = tf.nn.dropout(x,0.4)
		x = self.embedding(x)
		return x 


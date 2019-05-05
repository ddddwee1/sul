import tensorflow as tf 
import numpy as np 
import model3 as M 

class ResBlock_v1(M.Model):
	def initialize(self, outchn, stride):
		self.stride = stride
		self.outchn = outchn
		# self.bn1 = M.BatchNorm()
		self.c1 = M.ConvLayer(1, outchn//4, activation=M.PARAM_RELU, usebias=False, batch_norm=True)
		self.c2 = M.ConvLayer(3, outchn//4, pad='SAME_LEFT', stride=stride, activation=M.PARAM_RELU, usebias=False, batch_norm=True)
		self.c3 = M.ConvLayer(1, outchn, usebias=False, batch_norm=True)
		# shortcut 
		self.sc = M.ConvLayer(1, outchn, stride=stride, usebias=False, batch_norm=True)

	def build(self, input_shape):
		self.inchn = input_shape[-1]

	def forward(self, x):
		# res = self.bn1(x)
		res = x 
		res = self.c1(res)
		res = self.c2(res)
		res = self.c3(res)
		# shortcut 
		if self.inchn==self.outchn and self.stride==1:
			sc = x 
		else:
			sc = self.sc(x)
		res = res + sc 
		res = tf.nn.relu(res)
		return res 

class HeadBlock(M.Model):
	def initialize(self, outchn, stride=2):
		self.c1 = M.ConvLayer(7, outchn, stride=stride, pad='SAME_LEFT', usebias=False, activation=M.PARAM_RELU, batch_norm=True)
		self.p1 = M.MaxPool(3,stride=2, pad='VALID')

	def forward(self, x):
		x = self.c1(x)
		x = M.pad(x, 1)
		x = self.p1(x)
		return x

class ResNet(M.Model):
	def initialize(self, channel_list, blocknum_list):
		self.head = HeadBlock(channel_list[0])
		self.body = []
		for ii,(num, chn) in enumerate(zip(blocknum_list, channel_list[1:])):
			for i in range(num):
				self.body.append(ResBlock_v1(chn, 2 if (i==0 and ii>0) else 1))

	def forward(self, x):
		x = self.head(x)
		for block in self.body:
			x = block(x)
		return x 

class TransposeLayers(M.Model):
	def initialize(self, channel):
		self.t_conv1 = M.DeconvLayer(4,256,stride=2, activation=M.PARAM_RELU, batch_norm=True, usebias=False)
		self.t_conv2 = M.DeconvLayer(4,256,stride=2, activation=M.PARAM_RELU, batch_norm=True, usebias=False)
		self.t_conv3 = M.DeconvLayer(4,256,stride=2, activation=M.PARAM_RELU, batch_norm=True, usebias=False)
	def forward(self, x):
		x = self.t_conv1(x)
		x = self.t_conv2(x)
		x = self.t_conv3(x)
		return x 

class PosePredNet(M.Model):
	def initialize(self, num_pts):
		self.backbone = ResNet([64,256,512,1024, 2048], [3,4,6,3])
		self.head = TransposeLayers(256)
		self.lastlayer = M.ConvLayer(1, 17)
	def forward(self, x):
		x = self.backbone(x)
		x = self.head(x)
		x = self.lastlayer(x)
		return x 
	def norm(self,x):
		return x - np.array([[[123.68, 116.78, 103.94]]])
	def denorm(self,x):
		return x + np.array([[[123.68, 116.78, 103.94]]])

import model3 as M 
import layers3 as L 
import numpy as np 
import tensorflow as tf 

class SPConv(M.Model):
	def initialize(self, dw_chn, outchn, stride, activation=M.PARAM_LRELU):
		self.c1 = M.ConvLayer(1, dw_chn, activation=activation, batch_norm=True, usebias=False)
		self.c2 = M.DWConvLayer(3, 1, stride=stride, activation=activation, batch_norm=True, usebias=False)
		self.c3 = M.ConvLayer(1, outchn, batch_norm=True, usebias=False)
	def forward(self, x):
		x = self.c1(x)
		x = self.c2(x)
		x = self.c3(x)
		return x 

class ResUnit(M.Model):
	def initialize(self, numblk, dw_chn, outchn):
		self.blks = [SPConv(dw_chn, outchn, 1) for _ in range(numblk)]
	def forward(self, x):
		for c in self.blks:
			x2 = c(x)
			x = x + x2
		return x 

class MobileFaceHead(M.Model):
	def initialize(self, blk_list):
		self.c1 = M.ConvLayer(3, 64, stride=2, activation=M.PARAM_LRELU, batch_norm=True, usebias=False)
		self.R1 = ResUnit(blk_list[0], 64,64)
		self.c2 = SPConv(128, 64, 2)
		self.R2 = ResUnit(blk_list[1], 128, 64)
		self.c3 = SPConv(256, 128, 2)
		self.R3 = ResUnit(blk_list[2], 256, 128)
		self.c4 = SPConv(512, 128, 2)
		self.R4 = ResUnit(blk_list[3], 256, 128)
		self.c5 = M.ConvLayer(1, 512, activation=M.PARAM_LRELU, batch_norm=True, usebias=False)
		self.c6 = M.ConvLayer(7, 512, pad='VALID', batch_norm=True, usebias=False)
		self.fc1 = M.Dense(256)
	def forward(self, x, norm=True):
		if norm:
			x = x - 127.5
			x = x / 256.
		x = self.R1(self.c1(x))
		x = self.R2(self.c2(x))
		x = self.R3(self.c3(x))
		x = self.R4(self.c4(x))
		x = self.c6(self.c5(x))
		x = M.flatten(x)
		x = self.fc1(x)
		return x 

class Network(M.Model):
	def initialize(self, blk_list, num_classes):
		self.net = MobileFaceHead(blk_list)
		self.classifier = M.MarginalCosineLayer(num_classes)
	def forward(self, x, label):
		feat = self.net(x)
		logits = self.classifier(feat, label, 1.0, 0.5, 0.0)
		logits = logits*64
		return logits

net = Network([2, 8, 16, 4], 100)

# initialize network using dumb data
_ = np.random.random([1,112,112,3]).astype(np.float32) * 2. - 1.
lb_ = np.ones([1,100]).astype(np.float32)
_ = net(_, lb_)

# TO-DO: add training code


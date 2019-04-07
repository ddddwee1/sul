import tensorflow as tf 
import model3 as M 
import layers3 as L 
import numpy as np 

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
		self.c1 = M.ConvLayer(7, 64, pad='SAME_LEFT', stride=2, activation=M.PARAM_RELU, usebias=False, batch_norm=True)
		self.mp1 = M.MaxPool(3,2, pad='SAME_LEFT')
		self.block1 = ResBlock(256, 1, 3)
		self.block2 = ResBlock(512, 2, 4)
		self.block3 = ResBlock(1024, 2, 6)
		self.block4 = ResBlock(2048, 2, 3)

	def forward(self, x):
		x = self.c1(x)
		x = self.mp1(x)
		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)
		x = self.block4(x)
		return x 

net = ResNet()
saver = M.Saver(net)
saver.restore('./model/')

# perform numerical testing
tf.keras.backend.set_learning_phase(False)
x = np.ones([1,224,224,3]).astype(np.float32)
y = net(x).numpy()
y = np.transpose(y, [0,3,1,2])
print(y)
print(y.shape)

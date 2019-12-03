import wow 
import numpy as np 
import torch 

class Network(wow.Model):
	def initialize(self):
		self.c1 = wow.conv2D(5, 16)
	def forward(self, x):
		return self.c1(x)

net = Network()
xx = np.ones([1,3,16,16], dtype=np.float32)
xx = torch.from_numpy(xx)
yy = net(xx)
print(yy.shape)

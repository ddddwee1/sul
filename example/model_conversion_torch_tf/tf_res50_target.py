import tensorflow as tf 
import model3 as M 
import layers3 as L 
import numpy as np 

import pickle 
import glob

param_dict = {}
for f in glob.glob('./params/*.pkl'):
	name = f.replace('./params\\','').replace('.pkl','').replace('_modulesplit_','/').replace('_layersplit_','\\')

	# print(name)
	f = open(f,'rb')
	value = pickle.load(f)
	if len(value.shape)==4:
		value = np.transpose(value, (2,3,1,0))
	# print(value.shape)
	f.close()
	# input()
	param_dict[name] = value

def get_bn_values(name):
	mean = name+'\\running_mean'
	var = name+'\\running_var'
	gamma = name+'\\weight'
	beta = name+'\\bias'
	res = [mean, var, gamma, beta]
	res = [param_dict[i] for i in res]
	# print(res[0])
	# print(res[1])
	return res 

def get_conv_values(name):
	weight = name+'\\weight'
	bias = name+'\\bias'
	bias = None
	names = [weight, bias]
	# res = [param_dict[i] for i in res if not (i is None)]
	res = []
	for i in names:
		try:
			res.append(param_dict[i])
		except Exception as e:
			print(e)
			continue
	return res 

def get_layer_values(cname, bname):
	values = get_conv_values(cname) + get_bn_values(bname)
	# print(values[1].shape)
	return values

class ResUnit(M.Model):
	def initialize(self, out, name, stride, shortcut=False):
		self.shortcut = shortcut
		self.c1 = M.ConvLayer(1, out//4, usebias=False, activation=M.PARAM_RELU, batch_norm=True, values=get_layer_values(name+'/conv1', name+'/bn1'))
		self.c2 = M.ConvLayer(3, out//4, usebias=False, activation=M.PARAM_RELU, pad='SAME_LEFT', stride=stride, batch_norm=True, values=get_layer_values(name+'/conv2', name+'/bn2'))
		self.c3 = M.ConvLayer(1, out, usebias=False, batch_norm=True, values=get_layer_values(name+'/conv3', name+'/bn3'))
		if shortcut:
			self.sc = M.ConvLayer(1, out, usebias=False, stride=stride, batch_norm=True, values=get_layer_values(name+'/downsample/0', name+'/downsample/1'))

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
	def initialize(self, out, name, stride, num_units):
		self.units = []
		for i in range(num_units):
			self.units.append(ResUnit(out, '%s/%d'%(name,i), stride if i==0 else 1, True if i==0 else False))
	def forward(self, x):
		for unit in self.units:
			x = unit(x)
		return x 

class ResNet(M.Model):
	def initialize(self):
		self.c1 = M.ConvLayer(7, 64, pad='SAME_LEFT', stride=2, activation=M.PARAM_RELU, usebias=False, batch_norm=True, values=get_layer_values('//conv1', '//bn1'))
		self.mp1 = M.MaxPool(3,2, pad='SAME_LEFT')
		self.block1 = ResBlock(256, '//layer1', 1, 3)
		self.block2 = ResBlock(512, '//layer2', 2, 4)
		self.block3 = ResBlock(1024, '//layer3', 2, 6)
		self.block4 = ResBlock(2048, '//layer4', 2, 3)

	def forward(self, x):
		x = self.c1(x)
		x = self.mp1(x)
		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)
		x = self.block4(x)
		return x 

tf.keras.backend.set_learning_phase(False)
net = ResNet()
x = np.ones([1,224,224,3]).astype(np.float32)
y = net(x).numpy()
y = np.transpose(y, [0,3,1,2])
print(y)
print(y.shape)
# net.summary()
# net.c1.summary()
saver = M.Saver(net)
saver.save('./model/res50.ckpt')

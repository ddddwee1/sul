import network 
import numpy as np 
import tensorflow as tf 
import model3 as M 

def get_kernel(name, w):
	w.append(name+'/weights')

def getbn(name, bn1, bn2):
	bn1.append(name+'/BatchNorm/gamma')
	bn1.append(name+'/BatchNorm/beta')
	bn2.append(name+'/BatchNorm/moving_mean')
	bn2.append(name+'/BatchNorm/moving_variance')

def get_layer(name, w, bn, batch_norm=True):
	get_kernel(name, w)
	if batch_norm:
		getbn(name, w, bn)

def build_block(name, w, bn, sc=False):
	get_layer(name+'/conv1', w, bn)
	get_layer(name+'/conv2', w, bn)
	get_layer(name+'/conv3', w, bn)
	if sc:
		get_layer(name+'/shortcut', w, bn)

def get_net():
	w = []
	bn = []
	namebase = 'resnet_v1_50'

	get_layer(namebase+'/conv1',w,bn)

	unit = [3,4,6,3]
	for i in range(1,5):
		num_unit = unit[i-1]
		name_block = namebase + '/block%d'%i
		for j in range(1,num_unit+1):
			name_unit = name_block + '/unit_%d/bottleneck_v1'%j
			build_block(name_unit, w, bn, sc=j==1)
	return w+bn

def get_tconvs():
	w = []
	bn = []
	for i in range(1,4):
		get_layer('up%d'%i, w, bn)
	return w+bn

tf.keras.backend.set_learning_phase(False)

class PosePredNet(M.Model):
	def initialize(self, num_pts):
		self.backbone = network.ResNet([64,256,512,1024, 2048], [3,4,6,3])
		self.head = network.TransposeLayers(256)
		self.lastlayer = M.ConvLayer(1, 17)
	def forward(self, x):
		x = self.backbone(x)
		x = self.head(x)
		x = self.lastlayer(x)
		return x 

net = PosePredNet(17)

# backbone
mod = net.backbone
x = np.ones([1,256, 192, 3]).astype(np.float32)
y = mod(x)
a = mod.variables
for v in a:
	print(v.name, v.shape)
# restore params 
import pickle 
weights = pickle.load(open('vars.pkl','rb'))
weight_names = get_net()
weights = [[i,weights[i]] for i in weight_names]

for v,(vname,v_new) in zip(a, weights):
	print(v.name, vname, v.shape, v_new.shape)
	v.assign(v_new)

x = mod.forward_test(x)

# transpose 
mod_transpose = net.head
# x = np.ones([1,8, 6, 2048]).astype(np.float32)
y = mod_transpose(x)
a = mod_transpose.variables
for v in a:
	print(v.name, v.shape)
weights = pickle.load(open('vars.pkl','rb'))
weight_names = get_tconvs()
weights = [[i,weights[i]] for i in weight_names]

for v,(vname,v_new) in zip(a, weights):
	print(v.name, vname, v.shape, v_new.shape)
	v.assign(v_new)

x = mod_transpose(x)
# print(x[0,10,10])

lastlayer = net.lastlayer
y = lastlayer(x)
a = lastlayer.variables
for v in a:
	print(v.name, v.shape)
weights = pickle.load(open('vars.pkl','rb'))
weight_names = ['out/weights', 'out/biases']
weights = [[i,weights[i]] for i in weight_names]

for v,(vname,v_new) in zip(a, weights):
	print(v.name, vname, v.shape, v_new.shape)
	v.assign(v_new)

y = lastlayer(x)
print(y[0,10,10])


x = np.ones([1,256, 192, 3]).astype(np.float32)
y = net(x)
print(y[0,10,10])

saver = M.Saver(net)
saver.save('./posedetnet/posedetnet.ckpt')

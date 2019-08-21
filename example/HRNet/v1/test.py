import hrnet 
import numpy as np 
import tensorflow as tf 
import model3 as M 
import data_reader

class HRNET(M.Model):
	def initialize(self, num_pts):
		self.backbone = hrnet.ResNet()
		self.lastconv = M.ConvLayer(1, num_pts)
	def forward(self, x):
		feat = self.backbone(x)
		hmap = self.lastconv(feat)
		return hmap

def get_prediction(out, scale=4):
	shape = out.shape
	out = out.reshape([shape[0], shape[1]*shape[2], shape[3]])
	scr = np.max(out, axis=1)
	out = np.argmax(out, axis=1)

	res = np.zeros([shape[0], shape[3], 2])
	for i in range(shape[0]):
		for j in range(shape[3]):
			res[i,j,0] = out[i,j] % shape[1]
			res[i,j,1] = out[i,j] // shape[1]
	res = res * scale
	return res, scr

tf.keras.backend.set_learning_phase(False)
net = HRNET(19)

saver = M.Saver(net)
saver.restore('./model/')

# Read image
img = np.zeros([1,256,256,3]).astype(np.float32)
hmaps = net(img)
results = get_prediction(hmaps)[0]


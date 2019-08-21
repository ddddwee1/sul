import tensorflow as tf 
import model3 as M 
import numpy as np 
import hrnet
import layers3 as L 
import helper 

tf.keras.backend.set_learning_phase(False)

class FaceResNet(M.Model):
	def initialize(self):
		# self.resnet = mobilenet.MobileFaceHead([2, 8, 16, 4])
		# self.resnet = resnet.ResNet([64,64,128,256,512],[3,4,14,3],512)
		self.resnet = hrnet.ResNet(512)

	def forward(self, x):
		feat = self.resnet(x)
		# feat = tf.nn.dropout(feat, 0.4)
		return feat

model = FaceResNet()
saver = M.Saver(model)
saver.restore('./model/')
a = np.ones([1, 128, 128, 3]).astype(np.float32)
a = [a, helper.LayerName('inputholder')]
L.init_caffe_input(a)

a = model(a) # run once for initialization 
print(a[0])

f = open('abc.prototxt', 'w')
f.write(L.caffe_string)
f.close()
L.save_params('weights.pkl')

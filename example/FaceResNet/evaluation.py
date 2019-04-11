import tensorflow as tf 
import model3 as M 
import numpy as np 
import resnet
import cv2

class FaceResNet(M.Model):
	def initialize(self):
		self.resnet = resnet.ResNet([64,64,128,256,512], [3, 4, 14, 3], 512)

	def forward(self, x):
		feat = self.resnet(x)
		return feat

tf.keras.backend.set_learning_phase(True)

model = FaceResNet()
optimizer = tf.keras.optimizers.Adam(0.0001)
saver = M.Saver(model, optimizer)
saver.restore('./model/')
	
def extract_feature(imgname):
	img = cv2.imread(imgname)
	img = np.float32(img)[None,...]
	feat = model(img).numpy()[0]
	feat = feat.reshape([-1])
	feat = feat / np.linalg.norm(feat)
	return feat 

feat = extract_feature('1.jpg')

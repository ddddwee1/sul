import network 
import numpy as np 
import tensorflow as tf 
import model3 as M 
import cv2 
import matplotlib.pyplot as plt 

def get_prediction(out, scale=4):
	shape = out.shape
	out = out.reshape([shape[0], shape[1]*shape[2], shape[3]])
	scr = np.max(out, axis=1)
	out = np.argmax(out, axis=1)

	res = np.zeros([shape[0], shape[3], 2])
	for i in range(shape[0]):
		for j in range(shape[3]):
			res[i,j,0] = out[i,j] % shape[2]
			res[i,j,1] = out[i,j] // shape[1]
	res = res * scale
	return res, scr

def plot(pts):
	for i in range(len(pts)):
		x = pts[i,0]
		y = pts[i,1]
		plt.plot(x,y,'o')

tf.keras.backend.set_learning_phase(False)
net = network.PosePredNet(19)
saver = M.Saver(net)
saver.restore('./model/')

img2 = cv2.imread('00000000.jpg')
img = np.float32([img2])
out = net(img).numpy()
pts = get_prediction(out)[0]
# print(pts.shape)

plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plot(pts[0])
plt.show()

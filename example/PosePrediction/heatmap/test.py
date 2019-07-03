import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import network 
import numpy as np 
import model3 as M 
import tensorflow as tf 
import datareader
from tqdm import tqdm 
import cv2

net = network.PosePred(17)

reader = datareader.DataReader(16, processes=4, temp=20)

def visualize(hmap,lb,it):
	res = []
	for i in range(17):
		h = np.uint8(hmap[0,:,:,i]*255)
		res.append(h)
	res = np.stack(res,axis=2)
	res = np.amax(res, axis=2)
	# print(res.shape)
	cv2.imwrite('./vis/%d.jpg'%it, res)
	res = np.amax(lb[0], axis=2)
	res = np.uint8(res*255)
	cv2.imwrite('./vis/%d_lb.jpg'%it, res)


saver = M.Saver(net)
saver.restore('./model/')



batch = reader.get_next()
batch = np.float32(batch)


out = net(batch, 4, 0)
for i in range(19):
	visualize(tf.sigmoid(out[i]).numpy(), batch[i+1], i)


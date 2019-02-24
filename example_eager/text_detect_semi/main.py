import layers2 as L
# L.set_gpu('1')
import modeleag as M 
import tensorflow as tf 
import numpy as np 

import network_cls 
import network_rpn
import datareader

import cv2 

class Module(M.Model):
	def initialize(self):
		self.rpn_net = network_rpn.network()
		self.cls_net = network_cls.network()

	def forward(self, img, grid=16):
		# fix batch size at 1
		x = self.rpn_net(img)
		x = x[0]
		conf, xy, wh = x[:,:,0:1], x[:,:,1:3], x[:,:,3:]
		wh = tf.exp(wh) / 2.

		shape = xy.get_shape().as_list()
		
		patches = []

		coords = []

		for i in range(shape[0]):
			for j in range(shape[1]):
				# if conf[i,j,0].numpy()>0:
				if True:
					scale =  wh[i,j,1] *2 / 32.
					corner_x = j*grid + grid/2 - xy[i,j,0] - wh[i,j,0]
					corner_y = i*grid + grid/2 - xy[i,j,1] - wh[i,j,1]
					H = [scale, 0, corner_x, 0, scale, corner_y, 0, 0]
					out_shape = [32, int( wh.numpy()[i,j,0] * 2 / scale.numpy())]
					out = M.image_transform(img[0], H, out_shape, 'BILINEAR')
					out = tf.image.resize_images(out, [32, 100])
					patches.append(out)

					corner_x1 = corner_x + wh[i,j,0]*2
					corner_y1 = corner_y + wh[i,j,1]*2
					coords.append([corner_x.numpy(), corner_y.numpy(), corner_x1.numpy(), corner_y1.numpy()])


		cls_result = self.cls_net(patches)

		return cls_result, coords
		# return patches

	def get_patches(self, img, grid=16):
		img = tf.convert_to_tensor(img)
		x = self.rpn_net(img)
		x = x[0]
		conf, xy, wh = x[:,:,0:1], x[:,:,1:3], x[:,:,3:]
		wh = tf.exp(wh) / 2.

		shape = xy.get_shape().as_list()
		
		patches = []

		for i in range(shape[0]):
			for j in range(shape[1]):
				# if conf[i,j,0].numpy()>0:
				if True:
					scale =  wh[i,j,1] *2 / 32.
					corner_x = j*grid + grid/2 - xy[i,j,0] - wh[i,j,0]
					corner_y = i*grid + grid/2 - xy[i,j,1] - wh[i,j,1]
					H = [scale, 0, corner_x, 0, scale, corner_y, 0, 0]
					out_shape = [32, int( wh.numpy()[i,j,0] * 2 / scale.numpy())]
					out = M.image_transform(img[0], H, out_shape, 'BILINEAR')
					out = tf.image.resize_images(out, [32, 100])
					patches.append(out)
		return patches

def loss_func(img, model):
	with tf.GradientTape() as tape:
		cls_result = model([img])
		cls_result = tf.sigmoid(cls_result)
		ls = tf.reduce_mean(tf.square(cls_result - 1.))
	return ls, tape

def computeIOU(box1, box2):
	x1, y1, x2, y2 = box1
	x3, y3, x4, y4 = box2 
	dx = max(0, min(x2,x4) - max(x1,x3))
	dy = max(0, min(y2,y4) - max(y1,y3))
	inter = dx*dy 
	union = (x4 - x3) * (y4 - y3) + (x2 - x1) * (y2 - y1) - inter
	iou = inter / (union + 1)
	return iou 


if __name__=='__main__':

	mod = Module()
	optim = tf.train.AdamOptimizer(0.0001)

	saver = M.Saver(mod.rpn_net)
	saver.restore('./model_rpn/')

	img = cv2.imread('./img_180.jpg')

	####### magic ######
	h,w,_ = img.shape
	new_h = (h//32+1)*32
	new_w = (w//32+1)*32
	img = cv2.resize(img, dsize=(new_w, new_h))
	### end of magic ###

	img = np.float32(img)

	patches = mod.get_patches([img])
	print(len(patches))
	cnt = 0
	for i in patches:
		cnt +=1
		print(i.shape)
		i = np.uint8(i)
		cv2.imwrite('./result/%d.jpg'%cnt, i )

	# ls, tape = loss_func(img, mod)
	# grad = tape.gradient(ls, mod.variables)

	# print(grad[0])

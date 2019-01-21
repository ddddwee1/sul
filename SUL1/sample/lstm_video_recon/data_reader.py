import cv2
import numpy as np 
import random

class data_reader():
	def __init__(self):
		self.data = []
		print('Reading')
		for i in range(900):
			img = cv2.imread('./imgs/%d.jpg'%i)
			img = cv2.resize(img,(256,256))
			self.data.append(img)
		print('Read finished')

	def get_batch(self, bsize, step):
		src = []
		tgt = []
		for i in range(bsize):
			buf = []
			buf2 = []
			ind = random.randint(0, 900 - 2 - step)
			for j in range(step):
				buf.append(self.data[ind+j])
				buf2.append(self.data[ind+j+1])
			src.append(buf)
			tgt.append(buf2)
		src = np.array(src)
		tgt = np.array(tgt)
		# src should be in size [bsize, step, h, w, c]
		return src,tgt
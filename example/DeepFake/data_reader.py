import numpy as np 
import cv2 
import random
from multiprocessing.pool import ThreadPool
import glob 

def adjust_img(img):
	# a = np.random.randint(2)
	# if a==1:
	# 	img = np.flip(img, axis=1)
	# if random.random()>0.5:
	# 	img = np.flip(img, axis=1)
	img = cv2.resize(img, (64,64))
	img = np.float32(img) / 255.0
	return img 

def process(batch):
	# add more process here
	imgs, imgs2 = list(zip(*batch))
	imgs = [cv2.imread(i) for i in imgs]
	imgs = [adjust_img(i) for i in imgs]
	imgs2 = [cv2.imread(i) for i in imgs2]
	imgs2 = [adjust_img(i) for i in imgs2]
	batch = [np.float32(imgs), np.float32(imgs), np.float32(imgs2), np.float32(imgs2)]
	return batch

class data_reader():
	def __init__(self, bsize):
		self.bsize = bsize

		print('Fetching image list...')
		max_label = 0
		i1 = glob.glob('./res/*.*')
		i2 = glob.glob('./res2/*.*')
		self.data = [i1, i2]

		self.pool = ThreadPool(processes=1)
		self.prefetch()

	def prefetch(self):
		batch_i1 = random.sample(self.data[0], self.bsize)
		batch_i2 = random.sample(self.data[1], self.bsize)

		batch = list(zip(batch_i1, batch_i2))
		self.p = self.pool.apply_async(process, args=(batch, ))

	def get_next(self):
		batch = self.p.get()
		self.prefetch()
		return batch

if __name__=='__main__':
	a = cv2.imread('a.jpg')
	c = adjust_img(a)
	cv2.imshow('c',c)
	cv2.waitKey(0)

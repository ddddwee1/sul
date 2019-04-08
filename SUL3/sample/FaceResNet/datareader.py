import numpy as np 
import cv2 
import random
from multiprocessing.pool import ThreadPool

def adjust_img(img):
	# a = np.random.randint(2)
	# if a==1:
	# 	img = np.flip(img, axis=1)
	return img 

def process(batch, max_label):
	# add more process here
	imgs, labels = list(zip(*batch))
	imgs = [cv2.imread(i) for i in imgs]
	imgs = [adjust_img(i) for i in imgs]
	labels = np.eye(max_label+1)[np.array(labels)]
	batch = [np.float32(imgs), np.float32(labels)]
	return batch

class DataReader():
	def __init__(self, listfile, bsize):
		f = open(listfile, 'r')
		self.data = []
		print('Reading text file...')
		max_label = 0
		for line in f:
			line = line.strip().split('\t')
			img = line[1]
			label = int(line[2])
			if label>max_label:
				max_label = label
			self.data.append([img, label])
		random.shuffle(self.data)
		print('Finished')
		self.pos = 0
		self.epoch = 0
		self.bsize = bsize
		self.max_label = label
		self.iter_per_epoch = len(self.data)//self.bsize
		self.pool = ThreadPool(processes=1)
		self.prefetch()
		print('max_label:',max_label)

	def prefetch(self):
		if self.pos + self.bsize > len(self.data):
			self.pos = 0
			self.epoch += 1
			random.shuffle(self.data)

		batch = self.data[self.pos: self.pos+self.bsize]
		self.p = self.pool.apply_async(process, args=(batch, self.max_label))
		self.pos += self.bsize


	def get_next(self):
		batch = self.p.get()
		self.prefetch()
		return batch

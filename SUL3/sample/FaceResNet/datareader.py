import numpy as np 
import cv2 
import random

class DataReader():
	def __init__(self, listfile, bsize):
		f = open(listfile, 'r')
		self.data = []
		print('Reading text file...')
		max_label = 0
		for line in f:
			line = line.strip().split('\t')
			img = line[0]
			label = int(line[1])
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

	def adjust_img(self, img):
		return img 

	def process(self, batch):
		# add more process here
		imgs, labels = list(zip(batch))
		imgs = [cv2.imread(i) for i in imgs]
		imgs = [self.adjust_img(i) for i in imgs]
		labels = np.eye(self.max_label+1)[np.array(labels)]
		batch = [np.float32(imgs), np.float32(labels)]
		return batch

	def get_next(self):
		if self.pos + self.bsize > len(self.data):
			self.pos = 0
			self.epoch += 1
			random.shuffle(self.data)

		batch = self.data[self.pos: self.pos+self.bsize]
		batch = self.process(batch)
		self.pos += self.bsize

		return batch

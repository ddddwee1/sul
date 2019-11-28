import numpy as np 
import cv2 
import random
from multiprocessing.pool import ThreadPool
import time 

def adjust_img(img):
	# a = np.random.randint(2)
	# if a==1:
	# 	img = np.flip(img, axis=1)
	if random.random()>0.5:
		img = np.flip(img, axis=1)
	return img 

# def process(batch, eye):
# 	# add more process here
# 	imgs, labels = list(zip(*batch))
# 	# imgs = [cv2.resize(cv2.imread(i), (128,128)) for i in imgs]
# 	t = time.time()
# 	imgs = [cv2.imread(i) for i in imgs]
# 	t2 = time.time()
# 	print('DATA TIME', t2-t)
# 	imgs = [adjust_img(i) for i in imgs]
# 	t3 = time.time()
# 	print('FLIP TIME', t3-t2)
# 	labels = eye[np.array(labels)]
# 	batch = [np.float32(imgs), np.float32(labels)]
# 	t4 = time.time()
# 	print('CVT TIME', t4-t3)
# 	return batch

def process(sample):
	batch, eye = sample
	# add more process here
	img, label = batch
	# imgs = [cv2.resize(cv2.imread(i), (128,128)) for i in imgs]
	# t = time.time()
	img = cv2.resize(cv2.imread(img), (128,128))
	# t2 = time.time()
	# print('DATA TIME', t2-t)
	img = adjust_img(img)
	# t3 = time.time()
	# print('FLIP TIME', t3-t2)
	label = eye[label]
	# t4 = time.time()
	# print('CVT TIME', t4-t3)
	return img, label

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
		print(self.data[0])
		print('Finished')
		self.pos = 0
		self.epoch = 0
		self.bsize = bsize
		self.max_label = label
		self.iter_per_epoch = len(self.data)//self.bsize
		self.pool = ThreadPool(processes=32)
		self.eye = np.eye(self.max_label+1)
		self.prefetch()
		
		print('max_label:',max_label)

	def prefetch(self):
		if self.pos + self.bsize > len(self.data):
			self.pos = 0
			self.epoch += 1
			print(self.data[0])
			random.shuffle(self.data)

		batch = self.data[self.pos: self.pos+self.bsize]
		args = (batch, [self.eye]*len(batch))
		args = list(zip(*args))
		self.p = self.pool.map_async(process, args)
		self.pos += self.bsize


	def get_next(self):
		batch = self.p.get()
		batch = list(zip(*batch))
		batch = [np.float32(_) for _ in batch]
		self.prefetch()
		return batch

if __name__=='__main__':
	data_reader = DataReader('imglist_iccv_clean.txt', 256*4)
	for i in range(100):
		t1 = time.time()
		batch = data_reader.get_next()
		t2 = time.time()
		print(t2-t1)
	print(batch[0].shape)
	print(batch[1].shape)

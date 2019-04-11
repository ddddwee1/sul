import numpy as np 
import cv2 
import config 
from multiprocessing.pool import ThreadPool

LABEL_PATH =''
# add dynamic scaling and shifting later
SCALE_RANGE = [0.5, 1.5]
SHIFT_RANGE = [-100, 100]

def random_warp(img, label, scale_range, shift_range, imgsize):
	scale = np.random.uniform(low=scale_range[0], high=scale_range[1])
	trans_x = np.random.uniform(low=shift_range[0], high=shift_range[1])
	trans_y = np.random.uniform(low=shift_range[0], high=shift_range[1])
	H = np.float32([[scale, 0, trans_x], [0, scale, trans_y], [0,0,1]])
	dst = cv2.warpAffine(img, H, dsize=imgsize)
	label = label - np.float32([trans_x, trans_y, 0, 0])
	return dst, label

def filter_label(label, imgsize):
	res = []
	num_label = len(label)
	for i in range(num_label):
		buff = label[i]
		x1, y1, x2, y2 = buff[0] - buff[2]/2, buff[1] - buff[3]/2, buff[0] + buff[2]/2, buff[1] + buff[3]/2
		if not (x1<0 or x2>imgsize[0] or y1<0 or y2>imgsize[1]):
			res.append(buff)
	res = np.float32(res)
	return res 

def process(batch):
	imgs = []
	labs = []
	for fname, lab in batch:
		img = cv2.imread(imgm, 1)
		lab = np.float32(lab)
		lab_ = []
		while len(lab_)==0:
			img_, lab_ = random_warp(img, lab, SCALE_RANGE, SHIFT_RANGE, (512,512))
			lab_ = filter_label(lab_)
		imgs.append(img_)
		labs.append(lab_)
	batch = [np.float32(imgs), np.float32(labs)]
	return batch

class DataReader():
	def __init__(self, bsize):
		self.data = []
		for i in glob.glob(LABEL_PATH + '/*.txt'):
			i = i.replace('\\','/')
			# name = i.split('/')[-1]
			name = i.replace('label','imgs').replace('txt','jpg')
			label = self.read_label(i)
			self.data.append([name, label])

		self.pool = ThreadPool(processes=1)
		self.bsize = bsize
		self.pre_fetch()

	def read_label(self, file):
		res = []
		f = open(file)
		for i in f:
			i = i.strip()
			i = i.split('\t')
			bbox = [int(_) for _ in i[:4]]
			res.append(bbox)
		return res 

	def pre_fetch(self):
		batch = random.sample(self.data, self.bsize)
		self.p = self.pool.apply_async(process, args=(batch))

	def get_next(self):
		batch = self.p.get()
		self.pre_fetch()
		return batch

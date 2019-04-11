import numpy as np 
import cv2 
import glob
import config 
import random 
from multiprocessing.pool import ThreadPool

LABEL_PATH ='./label'
# add dynamic scaling and shifting later
SCALE_RANGE = [0.6, 1.4]
SHIFT_RANGE = [-200, 200]

def random_warp(img, label, scale_range, shift_range, imgsize):
	scale = np.random.uniform(low=scale_range[0], high=scale_range[1])
	trans_x = np.random.uniform(low=shift_range[0], high=shift_range[1])
	trans_y = np.random.uniform(low=shift_range[0], high=shift_range[1])
	H = np.float32([[scale, 0, trans_x*scale], [0, scale, trans_y*scale]])
	dst = cv2.warpAffine(img, H, dsize=imgsize)
	label = (label  + np.float32([trans_x, trans_y, 0, 0]) ) * scale
	return dst, label

def filter_label(label, category, imgsize):
	res = []
	cats = []
	num_label = len(label)
	for i in range(num_label):
		buff = label[i]
		# x1, y1, x2, y2 = buff[0] - buff[2]/2, buff[1] - buff[3]/2, buff[0] + buff[2]/2, buff[1] + buff[3]/2
		# if not (x1<0 or x2>imgsize[0] or y1<0 or y2>imgsize[1]):
		if not (buff[0]<0 or buff[0]>imgsize[0] or buff[1]<0 or buff[1]>imgsize[1]):
			res.append(buff)
			cats.append(category[i])
	res = np.float32(res)
	return res, cats

def compute_iou_simple(w1,h1,w2,h2):
	h = min(h1,h2)
	w = min(w1,w2)
	i = h*w 
	u = w1*h1 + w2 * h2 - i 
	iou = i / (u+1e-5)
	return iou 

def parse_single_label(labelmap, maskmap, lab, category):
	w,h = lab[2], lab[3]
	out_channel = len(config.anchor_shape) * len(config.anchor_scale)
	# determine which anchor
	shapes = config.anchor_shape
	scales = config.anchor_scale
	whs = []
	for s in shapes:
		for sc in scales:
			whs.append([sc*np.sqrt(s), sc/np.sqrt(s)])
	ious = np.float32([compute_iou_simple(w,h,w1,h1) for w1,h1 in whs])
	idx = np.argmax(ious)
	wh = whs[idx]

	# determine dx,dy
	x,y = int(lab[0]), int(lab[1])
	stride = config.stride
	col = x//stride
	row = y//stride
	dx = x - col * stride - 0.5 * stride
	dy = y - row * stride - 0.5 * stride

	# determine dw, dh
	dw = w / wh[0]
	dh = h / wh[1]
	dw = np.log(dw)
	dh = np.log(dh)
	xywh_idx = out_channel * 1 + idx*4

	# determine category (class)
	category_idx = out_channel * 5 + idx*config.categories + category

	# assign label map 
	labelmap[row, col, idx] = 1
	labelmap[row, col, xywh_idx:xywh_idx+4] = np.float32([dx,dy,dw,dh])
	labelmap[row, col, category_idx] = 1

	# assign mask map 
	maskmap[row, col, :out_channel] = 1
	maskmap[row, col, xywh_idx:xywh_idx+4] = 1
	maskmap[row, col, out_channel*5+idx*config.categories : out_channel*5+(idx+1)*config.categories] = 1

def parse_label(labelmap, maskmap, lab, categories):
	out_channel = len(config.anchor_shape) * len(config.anchor_scale) * ( 5 + config.categories)
	for i in range(len(lab)):
		parse_single_label(labelmap, maskmap, lab[i], categories[i])

def process(batch, label_size):
	imgs = []
	out_channel = len(config.anchor_shape) * len(config.anchor_scale) * ( 5 + config.categories)
	labs = np.zeros([len(batch), label_size[0], label_size[1], out_channel]).astype(np.float32)
	masks = np.zeros([len(batch), label_size[0], label_size[1], out_channel]).astype(np.float32)
	for i, (fname, lab, category) in enumerate(batch):
		img = cv2.imread(fname, 1)
		lab = np.float32(lab)
		# print(img.shape)
		if img.shape[0]>1000 and img.shape[1]>1000:
			img = cv2.resize(img, None, fx=0.5, fy=0.5)
			lab = lab * 0.5
		lab_ = []
		while len(lab_)==0:
			img_, lab_ = random_warp(img, lab, SCALE_RANGE, SHIFT_RANGE, (512,512))
			lab_, category_ = filter_label(lab_, category, config.image_size)
		imgs.append(img_)
		parse_label(labs[i], masks[i], lab_, category_)
	batch = [np.float32(imgs), labs, masks]
	return batch

def visualize(img, label):
	img = np.uint8(img)
	out_channel = len(config.anchor_shape) * len(config.anchor_scale)
	shapes = config.anchor_shape
	scales = config.anchor_scale
	stride = config.stride
	whs = []
	for s in shapes:
		for sc in scales:
			whs.append([sc*np.sqrt(s), sc/np.sqrt(s)])
	shape_label = label.shape
	for r in range(shape_label[0]):
		for c in range(shape_label[1]):
			for idx in range(out_channel):
				if label[r,c,idx] > 0:
					xywh = label[r,c,out_channel+idx*4:out_channel+idx*4+4]
					x = c*stride + 0.5*stride + xywh[0]
					y = r*stride + 0.5*stride + xywh[1]
					w = np.exp(xywh[2]) * whs[idx][0]
					h = np.exp(xywh[3]) * whs[idx][1]
					x1,y1,x2,y2 = int(x-0.5*w) , int(y-0.5*h), int(x+0.5*w), int(y+0.5*h)
					cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
	return img 

class DataReader():
	def __init__(self, bsize):
		print('Loading meta files...')
		self.data = []
		for i in glob.glob(LABEL_PATH + '/*.txt'):
			i = i.replace('\\','/')
			# name = i.split('/')[-1]
			name = i.replace('label','imgs').replace('.txt','')
			label, classes = self.read_label(i)
			self.data.append([name, label, classes])

		self.pool = ThreadPool(processes=1)
		self.bsize = bsize
		print('Meta files loaded.')
		self.pre_fetch()

	def read_label(self, file):
		res = []
		classes = []
		f = open(file)
		for i in f:
			i = i.strip()
			i = i.split('\t')
			bbox = [float(_) for _ in i[:4]]
			classes.append(int(i[4]))
			res.append(bbox)
		res = np.float32(res)
		return res, classes

	def pre_fetch(self):
		batch = random.sample(self.data, self.bsize)
		self.p = self.pool.apply_async(process, args=(batch, config.output_shape))

	def get_next(self):
		batch = self.p.get()
		self.pre_fetch()
		return batch

if __name__=='__main__':
	reader = DataReader(1)
	img, label, mask = reader.get_next()
	print(img.shape)
	print(label.shape)
	img = img[0]
	label = label[0]
	img = visualize(img, label)
	cv2.imshow('a', img)
	cv2.waitKey(0)

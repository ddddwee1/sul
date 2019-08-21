import numpy as np 
import json 
import random
from multiprocessing.pool import ThreadPool
import simple_geometry
import cv2 
import scipy.stats as st

def generate_gaus(size):
	nsig = size/3
	kernlen = size
	interval = (2*nsig+1.)/(kernlen)
	x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
	kern1d = np.diff(st.norm.cdf(x))
	kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
	kernel = kernel_raw/kernel_raw.sum()
	kernel = kernel / kernel.max()
	return kernel

def draw_heat_map(annot, hmap, gaus_size):
	HEIGHT, WIDTH = hmap.shape[:2]
	canvas = np.zeros([HEIGHT + 100, WIDTH + 100]).astype(np.float32)
	xc, yc = annot
	active = generate_gaus(gaus_size*2+1)
	canvas[yc-gaus_size + 50:yc+gaus_size+1 + 50 , xc-gaus_size + 50: xc+gaus_size+1 + 50] = active
	hmap[:] = canvas[50:-50, 50:-50]

def draw_gaussian(label, size, down_sample):
	res = np.zeros([size[0], size[1], len(label)])
	for i in range(len(label)):
		l = [label[i][0]//down_sample, label[i][1]//down_sample]
		draw_heat_map(l, res[:,:,i], 3)
	return res 

def random_shift(img, label):
	xs = [i[0] for i in label]
	ys = [i[1] for i in label]
	minx = - np.array(xs).min()
	maxx = 255 - np.array(xs).max()
	miny = - np.array(ys).min()
	maxy = 255 - np.array(ys).max()
	tx = random.random() * (maxx - minx) + minx
	ty = random.random() * (maxy - miny) + miny 
	H = np.float32([[1,0,tx], [0,1,ty]])
	img = cv2.warpAffine(img, H, (256,256))
	label = np.float32(label) + np.float32([tx, ty])
	label = label.astype(int).tolist()
	return img, label

def process(data):
	batch = []
	for i in data:
		img = cv2.imread(i.replace('.json',''))
		label = json.load(open(i))

		#random scale 
		scale = random.random() * 0.1 + 0.95
		H = np.float32([[scale, 0, 0], [0, scale, 0]])
		img = cv2.warpAffine(img, H, img.shape)
		label = np.float32(label) * scale

		D = json.load(open(i.replace('.json','_D.json')))
		# test occlusion
		# print(label)
		# check valid label 
		try:
			pts3d = np.concatenate([np.float32(label).reshape([-1,2]), np.float32(D).reshape([-1,1])], axis=1)
			mask = simple_geometry.test_occlusion(pts3d)[:,0]
		except:
			continue
		# draw gassuian map
		# random_shift 
		img, label = random_shift(img, label)
		# print(label)
		label = draw_gaussian(label, [64,64], down_sample=4)
		# apply occlusion mask
		label = label * mask *255
		
		batch.append([img, label])
	batch = list(zip(*batch))
	batch = [np.float32(_) for _ in batch]
	return batch

class data_reader():
	def __init__(self, bsize):
		self.data = []
		print('Reading list...')
		f = open('list_small.txt')
		for i in f:
			i = i.strip()
			self.data.append(i)

		random.shuffle(self.data)
		print('List loaded')
		self.bsize = bsize
		self.pool = ThreadPool(processes=1)
		self.prefetch()

	def prefetch(self):
		batch = random.sample(self.data, self.bsize)
		self.p = self.pool.apply_async(process, args=(batch,))

	def get_next(self):
		batch = self.p.get()
		self.prefetch()
		return batch

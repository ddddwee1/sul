import numpy as np 
import random
import cv2 
import scipy.stats as st
import pickle 
import model3 as M 
import config 

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
	canvas = np.zeros([HEIGHT + 200, WIDTH + 200]).astype(np.float32)
	xc, yc = annot
	if xc<0:
		xc = 0
	if yc<0:
		yc = 0
	active = generate_gaus(gaus_size*2+1)
	canvas[yc-gaus_size + 100:yc+gaus_size+1 + 100 , xc-gaus_size + 100: xc+gaus_size+1 + 100] = active
	hmap[:] = canvas[100:-100, 100:-100]

def draw_gaussian(label, size, down_sample):
	res = np.zeros([size[0], size[1], len(label)])
	for i in range(len(label)):
		l = [label[i][0]//down_sample, label[i][1]//down_sample]
		draw_heat_map(l, res[:,:,i], 3)
	return res 

def random_shift(img, label):
	xs = [i[0] for i in label if i[2]>0]
	ys = [i[1] for i in label if i[2]>0]
	minx = - np.array(xs).min() +10
	maxx = 245 - np.array(xs).max()
	miny = - np.array(ys).min() +10
	maxy = 245 - np.array(ys).max()
	tx = random.random() * (maxx - minx) + minx
	ty = random.random() * (maxy - miny) + miny 
	H = np.float32([[1,0,tx], [0,1,ty]])
	img = cv2.warpAffine(img, H, (256,256))
	label = np.float32(label) + np.float32([tx, ty,0])
	return img, label

def draw_label(pts):
	# print(pts.shape)
	label = pts[:,:2]
	label = np.int32(label)
	label = draw_gaussian(label, [64,64], down_sample=4)

	mask = pts[:,2]
	mask[mask>0] = 1
	# mask = 1
	label = label * mask * 255
	return label, mask

def crop(img, bbox, pts):
	l = int(bbox[2] - bbox[0])
	H = np.float32([[1, 0, -bbox[0]], [0, 1, -bbox[1]]])
	img = cv2.warpAffine(img, H, (l,l))
	img = cv2.resize(img, (256,256))
	pts = (pts - np.float32([bbox[0], bbox[1], 0])) * 256 / l 

	# print('aa', type(pts))
	scale = random.random() * 0.1 + 0.95
	H = np.float32([[scale, 0, 0], [0, scale, 0]])
	img = cv2.warpAffine(img, H, (256,256))
	pts = pts * scale 
	# print('aa', type(pts))
	try:
		img, pts = random_shift(img, pts)
	except:
		abc = 111
	# print('aa', type(pts))
	return img, pts

def wipe_invalid(data):
	res = []
	for d in data:
		box = d['bbox']
		l = box[2] - box[0]
		if l>50:
			res.append(d)
	return res 

class data_reader(M.ThreadReader):
	def _get_data(self):
		result = []
		for dt in config.datasets:
			data = pickle.load(open('%s.pkl'%dt,'rb'))
			data = wipe_invalid(data)
			result += data 
		# data = pickle.load(open('coco_pts.pkl', 'rb'))
		# data2 = pickle.load(open('mpii_pts.pkl', 'rb'))
		# data2 = wipe_invalid(data2)
		# data = data + data2
		print('DATA LENGTH:',len(result))
		return result
	def _next_iter(self):
		batch = random.sample(self.data, self.bsize)
		# print(batch)
		return batch

	def _process_data(self, item):
		img = item['image_name']
		bbox = item['bbox']
		pts = item['keypoints']
		# print(img)
		img = cv2.imread(img)
		# print(type(img))
		# print(type(pts))
		img, pts = crop(img, bbox, pts)
		# print(type(pts))
		label, mask = draw_label(pts)
		# mask = mask[None,...]
		# mask = mask[None,...]
		return img, label, mask

	def _post_process(self, x):
		x = list(zip(*x))
		x = [np.float32(_) for _ in x]
		# normalize data 
		x[0] = (x[0] - np.float32([0.485, 0.456, 0.406]) ) / np.float32([0.229, 0.224, 0.225])
		return x 


def plot_hmap(hmap):
	hmap = np.amax(hmap, axis=-1)
	hmap = np.uint8(hmap)
	return hmap

if __name__=='__main__':
	reader = data_reader(1)
	for i in range(100):
		aa = reader.get_next()
		print(len(aa))
		imgs, dts = aa 
		img = np.uint8(imgs[0])
		hmap = plot_hmap(dts[0])
		cv2.imshow('a',img)
		cv2.imshow('b', hmap)
		cv2.waitKey(0)

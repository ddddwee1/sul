import numpy as np 
import random
import cv2 
import scipy.stats as st
import pickle 
import config 
from tqdm import tqdm 

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
	if xc<0:
		xc = 0
	if yc<0:
		yc = 0
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

dt = pickle.load(open('coco_pts_refine.pkl', 'rb'))

dt_refine = []

bar = tqdm(range(len(dt)))
Num_invalid = 0
for i in bar:
	item = dt[i]
	try:
		img = item['image_name']
		bbox = item['bbox']
		pts = item['keypoints']
		img = cv2.imread(img)
		img, pts = crop(img, bbox, pts)
		label, mask = draw_label(pts)
		dt_refine.append(item)
	except:
		Num_invalid+=1
		print(Num_invalid)
pickle.dump(dt_refine, open('coco_pts_refined.pkl', 'wb'))

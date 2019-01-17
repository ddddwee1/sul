import cv2 
import numpy as np 
import random 
import scipy.stats as st

WIDTH = 512
HEIGHT = 512
SCALE_RANGE = [0.05, 0.2]

def get_data():
	data = []
	print('Reading data...')
	f = open('annotation.txt')
	for i in f:
		i = i.strip()
		i = i.split('\t')
		fname = i[0]
		coord = [float(t) for t in i[1:]]
		x1 = coord[0::4]
		y1 = coord[1::4]
		x2 = coord[2::4]
		y2 = coord[3::4]
		coord = list(zip(x1,y1,x2,y2))
		coord = [[(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1] for x1,y1,x2,y2 in coord]
		img = cv2.imread(fname)
		data.append([img, coord])
	print('Finish')
	return data

def random_crop(img,annot):
	# right btm corner
	x2s = [i[0]+i[2]/2 for i in annot]
	y2s = [i[1]+i[3]/2 for i in annot]

	# left top corner
	x1s = [i[0]-i[2]/2 for i in annot]
	y1s = [i[1]-i[3]/2 for i in annot]

	# get the shift range
	xmin = np.max(np.array(x2s)) - WIDTH
	xmax = np.min(np.array(x1s))
	ymin = np.max(np.array(y2s)) - HEIGHT
	ymax = np.min(np.array(y1s))
	# get transform value
	x_trans = random.random()*(xmax-xmin) + xmin
	y_trans = random.random()*(ymax-ymin) + ymin
	# get transformation matrix and do transform
	# print(xmin,xmax)
	# x_trans = y_trans = 0
	M = np.float32([[1,0,-x_trans],[0,1,-y_trans]])

	img_result = img.copy()
	img_result = cv2.warpAffine(img_result,M,(WIDTH,HEIGHT))
	# substract the transformed pixels
	annot = np.float32(annot) - np.float32([[x_trans,y_trans,0,0]])
	# print(annot)
	return img_result,annot.tolist()

def random_scale(img,annot):
	# set scale range
	scale_range = SCALE_RANGE
	annot = np.float32(annot)
	scale = random.random()*(scale_range[1]-scale_range[0])+scale_range[0]
	# scaling the annotation and image
	annot = annot * scale
	img_result = cv2.resize(img,None,fx=scale,fy=scale)
	return img_result , annot.tolist()

def process_img(img, annot):
	img,annot = random_scale(img, annot)
	img,annot = random_crop(img, annot)
	return img, annot

def get_heat_map(annot):
	canvas = np.zeros([HEIGHT//4 + 100, WIDTH//4 + 100])
	for c in annot:
		xc, yc = int(c[0]/4)+50 , int(c[1]/4)+50
		size = int(min(c[2],c[3])) //8
		active = generate_gaus(size*2+1)
		# print(active.shape)
		# print(xc-size, xc+size+1, yc-size, yc+size+1)
		canvas[yc-size:yc+size+1 , xc-size: xc+size+1] = np.maximum(active, canvas[yc-size:yc+size+1 , xc-size: xc+size+1])
	canvas = canvas[:,:,np.newaxis]
	canvas = canvas[50:-50, 50:-50]
	return canvas

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

def draw(img, co, wait=0):
	img = img.copy()
	for c in co:
		x1,y1,x2,y2 = int(c[0]-c[2]/2), int(c[1]-c[3]/2), int(c[0]+c[2]/2), int(c[1]+c[3]/2)
		cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
	cv2.imshow('img', img)
	cv2.waitKey(wait)

def process(bundle):
	print('single_img')
	img, co = bundle
	img_, co_ = process_img(img, co)
	hmap = get_heat_map(co_)
	return img_, hmap

def post_process(bundle):
	return list(zip(*bundle))

import random
import cv2 
import numpy as np 
import glob 

def load_annot(fname):
	# annot : [x1, y1, x2, y2]
	f = open(fname)
	annots = []
	for i in f:
		i = i.strip()
		i = i.split(',')
		annot = [int(i[0]), int(i[1]), int(i[2]), int(i[3])]
		if annot[2]-annot[0]<=1 or annot[3]-annot[1]<=1:
			continue
		else:
			annots.append(annot)
	return annots

def load_data():
	data = []
	for f in glob.glob('./task1/train/img/*.*'):
		f = f.replace('\\','/') # windows use \ instead of /
		img = cv2.imread(f)
		annot = load_annot(f.replace('.jpg','.txt').replace('/img/','/gt/'))
		data.append([img, annot])
	print('Data size:',len(data))
	return data

def random_crop(img, annot):
	xy_minmax = [9999, 9999, -9999, -9999]
	for a in annot:
		if a[0]<xy_minmax[0]:
			xy_minmax[0] = a[0]
		if a[1]<xy_minmax[1]:
			xy_minmax[1] = a[1]
		if a[2]>xy_minmax[2]:
			xy_minmax[2] = a[2]
		if a[3]>xy_minmax[3]:
			xy_minmax[3] = a[3]

	h,w,_ = img.shape
	crop_x1 = random.randint(0, xy_minmax[0]-1)
	crop_x2 = random.randint(xy_minmax[2]+1, w-1)
	crop_y1 = random.randint(0, xy_minmax[1]-1)
	crop_y2 = random.randint(xy_minmax[3]+1, h-1)

	cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]

	annot = np.array(annot) - np.array([[crop_x1, crop_y1, crop_x1, crop_y1]])
	return cropped_img ,annot

def random_scale(img, annot, size_range):
	hws = [[i[3]-i[1], i[2]-i[0]] for i in annot]
	hs = [i[0] if i[0]<i[1] else i[1] for i in hws]

	hmax = np.array(hs).max()
	hmin = np.array(hs).min()

	scale_range_1 = size_range[0]/hmin
	scale_range_2 = size_range[1]/hmax

	scales = [min(scale_range_1, scale_range_2), max(scale_range_1, scale_range_2)]
	scale = random.random() * (scales[1]-scales[0]) + scales[0]

	img = cv2.resize(img, None, fx=scale, fy=scale)

	annot = np.float32(annot) * scale
	return img, annot

def random_translate(img, annot, size):
	canvas = np.zeros([size, size, 3], dtype=np.uint8)

	xy_minmax = [9999, 9999, -9999, -9999]
	for a in annot:
		if a[0]<xy_minmax[0]:
			xy_minmax[0] = a[0]
		if a[1]<xy_minmax[1]:
			xy_minmax[1] = a[1]
		if a[2]>xy_minmax[2]:
			xy_minmax[2] = a[2]
		if a[3]>xy_minmax[3]:
			xy_minmax[3] = a[3]

	# w/h cannot exceed the 0.5 * img_w/h

	hws = [[i[3]-i[1], i[2]-i[0]] for i in annot]
	hws = [max(h,w) if (h>0.5*size or w>0.5*size) else 0 for h,w in hws]
	if sum(hws)!=0:
		# apply re-scale
		scale = 0.5 * size / max(hws)
		img = cv2.resize(img, None, fx=scale, fy=scale)
		annot = np.float32(annot) * scale
		xy_minmax = np.float32(xy_minmax) * scale

	x_trans = [-xy_minmax[0], size-1-xy_minmax[2]]
	y_trans = [-xy_minmax[1], size-1-xy_minmax[3]]

	x_trans = random.random() * (x_trans[1]-x_trans[0]) + x_trans[0]
	y_trans = random.random() * (y_trans[1]-y_trans[0]) + y_trans[0]

	warp_mtx = np.float32([[1,0,x_trans],[0,1,y_trans]])
	img_res = cv2.warpAffine(img, warp_mtx, (size,size))

	annot = np.float32(annot) + np.float32([[x_trans, y_trans, x_trans, y_trans]])

	return img_res, annot

def process_image(img, annot):
	img, annot = random_crop(img, annot)
	img, annot = random_scale(img, annot, [20, 60])
	img, annot = random_translate(img, annot, 512)
	return img, annot

def draw(img, annot, color=(0,255,0)):
	img = np.uint8(img.copy())
	for a in annot:
		cv2.rectangle(img, (int(a[0]), int(a[1])), (int(a[2]), int(a[3])), color, 2)
	return img 

def annot_to_grid(img, annot, grid):
	h,w,_ = img.shape
	sizeh = h//grid
	sizew = w//grid

	# initialize 
	conf = np.zeros([sizeh, sizew, 1], dtype=np.float32)
	xywh = np.ones([sizeh, sizew, 4], dtype=np.float32)
	
	for a in annot:
		x,y,w,h = (a[0]+a[2])/2, (a[1]+a[3])/2, a[2]-a[0], a[3]-a[1]
		# assign conf_map 
		row = int(y//grid)
		col = int(x//grid)
		conf[row,col] = 1.

		dx, dy = x-col*grid-grid//2, y-row*grid-grid//2
		xywh[row,col,0] = dx
		xywh[row,col,1] = dy
		xywh[row,col,2] = np.log(w) 
		xywh[row,col,3] = np.log(h)

	return img, conf, xywh

def grid_to_annot(conf, xywh, grid, thresh=0.):
	r,c,_ = conf.shape
	annots = []
	for i in range(r):
		for j in range(c):
			if conf[i][j][0]>thresh:
				x = xywh[i][j][0] + j*grid + grid//2
				y = xywh[i][j][1] + i*grid + grid//2
				w = np.exp(xywh[i][j][2])
				h = np.exp(xywh[i][j][3])
				annot = [x-w/2, y-h/2, x+w/2, y+h/2]
				annots.append(annot)
	return annots

if __name__=='__main__':
	img = cv2.imread('img_7.jpg')
	annot = load_annot('gt_img_7.txt')
	img, annot = process_image(img, annot)
	img, conf, xywh = annot_to_grid(img, annot, 16)
	annot = grid_to_annot(conf, xywh, 16)
	img = draw(img, annot)
	cv2.imshow('img',img)
	cv2.waitKey(0)

from PIL import Image
import numpy as np 
import cv2
import glob 
import random
import config 

scale_range = config.scale_range
shift_range = config.shift_range
categories = config.categories
imgsize = config.imgsize
down_rate = config.down_rate

def read_label(path):
    im = Image.open(path)
    img = np.array(im,np.uint8)
    return img

def process_img(img, lab):
	# print(img.shape)
	scale = np.random.uniform(low=scale_range[0], high=scale_range[1])
	trans_x = np.random.uniform(low=shift_range[0], high=shift_range[1])
	trans_y = np.random.uniform(low=shift_range[0], high=shift_range[1])
	H = np.float32([[scale, 0, trans_x*scale], [0, scale, trans_y*scale]])
	# print(imgsize)
	dst = cv2.warpAffine(img, H, dsize=imgsize)
	lab = cv2.warpAffine(lab, H, dsize=imgsize, flags=cv2.INTER_NEAREST)
	lab = np.eye(categories)[lab]
	return dst, lab 

def process(batch):
	img, lab = list(zip(*batch))
	res = []
	for i in range(len(img)):
		im, lb = process_img(img[i], lab[i])
		res.append([im, lb])
	res = list(zip(*res))
	res = [np.float32(_) for _ in res]
	return res 

class DataReader():
	def __init__(self, bsize):
		self.bsize = bsize
		print('Reading data...')
		self.data = []
		for i in glob.glob('./annos_2ch/*.png'):
			i = i.replace('\\','/')
			img = cv2.imread(i.replace('annos_2ch', 'images').replace('png', 'jpg'))
			lab = read_label(i)
			lab = lab[:,:,1]

			laplacian = cv2.Laplacian(lab,cv2.CV_64F)
			laplacian = np.uint8(laplacian)
			laplacian[laplacian>0] = 1
			kernel = np.ones((3,3),np.uint8)
			laplacian = cv2.dilate(laplacian,kernel,iterations = 1)
			self.data.append([img, laplacian])

	def get_next(self):
		batch = random.sample(self.data, self.bsize)
		batch = process(batch)
		return batch

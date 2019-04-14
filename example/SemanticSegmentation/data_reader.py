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

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([b,g,r])

    cmap = cmap/255 if normalized else cmap
    return cmap

def process_img(img, lab, mask):
	# print(img.shape)
	scale = np.random.uniform(low=scale_range[0], high=scale_range[1])
	trans_x = np.random.uniform(low=shift_range[0], high=shift_range[1])
	trans_y = np.random.uniform(low=shift_range[0], high=shift_range[1])
	H = np.float32([[scale, 0, trans_x*scale], [0, scale, trans_y*scale]])
	# print(imgsize)
	dst = cv2.warpAffine(img, H, dsize=imgsize)
	lab = cv2.warpAffine(lab, H, dsize=imgsize, flags=cv2.INTER_NEAREST)
	lab = np.eye(categories)[lab]
	mask = cv2.warpAffine(mask, H, dsize=imgsize, flags=cv2.INTER_NEAREST)
	# mask = mask[..., None]
	return dst, lab, mask 

def process(batch):
	img, lab, mask = list(zip(*batch))
	res = []
	for i in range(len(img)):
		im, lb, msk = process_img(img[i], lab[i], mask[i])
		res.append([im, lb, msk])
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
			lab = lab[:,:,0]
			lab = lab - 1
			mask = np.ones(lab.shape).astype(np.uint8)
			mask[lab==255] = 0
			lab = lab * mask
			self.data.append([img, lab, mask])

	def get_next(self):
		batch = random.sample(self.data, self.bsize)
		batch = process(batch)
		return batch

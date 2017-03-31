import cv2
import numpy as np 

PARAM_GRAY = 0
PARAM_COLOR = 1

def fromListGetImages(listFile,gray=1,shape=None,resize=None):
	print('getting images from list file',listFile,'...')
	valid = False
	if (gray==1 and shape[-1]==3) or (gray==0 and shape[-1]==1):
		valid = True
	assert valid
	f = open(listFile)
	pics = []
	dirs = []
	for line in f:
		dirs.append(line.replace('\n',''))
	checkpoint = len(dirs)//100
	for i in range(len(dirs)):
		if (i+1)%checkpoint==0:
			print('progress:',str(i//checkpoint)+'%')
		a = cv2.imread(dirs[i],gray)
		if resize!=None:
			a = cv2.resize(a,(resize,resize))
		pics.append(a)
	pics = np.float32(pics)
	if shape!=None:
		pics = pics.reshape(shape)
	return pics

def normalizeImgs(pics,minval=-1.0,maxval=1.0):
	rg = maxval-minval
	pics = pics*rg/255.0
	pics = pics+minval
	return pics

def originalImgs(pics,minval=-1.0,maxval=1.0):
	rg = maxval-minval
	pics = pics-minval
	pics = pics*255.0/rg
	pics = np.floor(pics).astype(np.uint8)
	return pics
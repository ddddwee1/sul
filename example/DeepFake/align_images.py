import sys
sys.path.append('./util_align/')
import face_model
import argparse
import cv2
import numpy as np
import os
import glob
import cv2

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--path', type=str, help='image folder')

args = parser.parse_args()
model = face_model.FaceModel(args)

print('start')
# get gallery list
gallery_dir = args.path
if os.name == 'nt':
	gallery_paths = glob.glob('%s\\**\\*.*'%gallery_dir,recursive=True)
else:
	gallery_paths = glob.glob('%s/**/*.*'%gallery_dir,recursive=True)

#aligh for gallery list
if not os.path.exists('./res/'):
	os.mkdir('./res/')

cnt_processed = 0
for line in gallery_paths:
	print(line)
	img_raw1=cv2.imread(line,cv2.IMREAD_COLOR)
	bbox,points=model.get_faces(img_raw1)
	if img_raw1 is not None:
		if bbox is not None and bbox.shape[0]!=0:
			for i in range(bbox.shape[0]):
				point= points[i,:].reshape((2,5)).T
				score= bbox[i,4]
				processed=model.get_input2(img_raw1,point)
				cv2.imwrite('./res/%d.jpg'%cnt_processed, processed)
				cnt_processed += 1

import cv2
import numpy as np 
import os 
import time 

def extract_frames(fname,prefix,skip=1):
	print('Extracting %s'%fname)
	t1 = time.time()
	cap = cv2.VideoCapture(fname)
	cnt = 0
	images = []
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break
		if cnt%skip==0:
			if prefix is None:
				images.append(frame)
			else:
				imgname = prefix+'_%08d.jpg'%cnt
				if not os.path.exists(imgname):
					cv2.imwrite(imgname, frame)
		cnt += 1
	t2 = time.time()
	print('Extraction finished. Time elapsed:', t2-t1)
	return images

class video_saver():
	def __init__(self,name,size, frame_rate=15.0):
		self.name = name
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		self.vidwriter = cv2.VideoWriter(name,fourcc,frame_rate,(size[1],size[0]))
	def write(self,img):
		self.vidwriter.write(img)
	def finish(self):
		self.vidwriter.release()

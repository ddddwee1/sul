import cv2
import numpy as np 

def extract_frames(fname,prefix,skip=1):
	print('Extracting %s'%fname)
	cap = cv2.VideoCapture(fname)
	cnt = 0
	imgnames = []
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break
		if cnt%skip==0:
			imgname = prefix+'_%d.jpg'%cnt
			if not os.path.exists(imgname):
				cv2.imwrite(imgname, frame)
			imgnames.append(imgname)
		cnt += 1
	return imgnames

class video_saver():
	def __init__(self,name,size, frame_rate=15.0):
		self.name = name
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		self.vidwriter = cv2.VideoWriter(name,fourcc,frame_rate,(size[1],size[0]))
	def write(self,img):
		self.vidwriter.write(img)
	def finish(self):
		self.vidwriter.release()

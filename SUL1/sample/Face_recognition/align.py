import numpy as np 
import cv2 



def draw_fun(event,x,y,flags,param):
	global lmklist,img2
	if event==cv2.EVENT_LBUTTONUP:
		cv2.circle(img2,(x,y),5,(0,0,255),-1)
		lmklist.append([x,y])
		print(lmklist)

def draw_2p(img):
	global img2,lmklist
	lmklist = []
	img2 = img.copy()
	cv2.namedWindow('img')
	cv2.setMouseCallback('img',draw_fun)
	while True:
		cv2.imshow('img',img2)
		k = cv2.waitKey(20)
		if k==ord('s'):
			cv2.destroyAllWindows()
			return np.float32(lmklist)
		if k==ord('r'):
			img2 = img.copy()
			lmklist = []

def rotate_2p(img,lmk):
	eye_c = lmk[0]
	mse_c = lmk[1]
	lmk2 = lmk-mse_c.reshape([-1,2])
	kaku = eye_c - mse_c
	kaku = np.arctan(-kaku[0]/kaku[1])/np.pi*180.0
	print(kaku)
	rows,cols,_ = img.shape
	M = cv2.getRotationMatrix2D((int(mse_c[0]),int(mse_c[1])),kaku,1)
	dst = cv2.warpAffine(img,M,(cols,rows))
	kaku = kaku*np.pi / 180.0
	m2 = np.float32([[np.cos(kaku),-np.sin(kaku)],[np.sin(kaku),np.cos(kaku)]])
	lmk2 = lmk2.dot(m2)
	lmk2 += mse_c
	return dst,lmk2

def get_2p(img,lmk,path):
	M = np.float32([[1,0,2000],[0,1,2000]])
	rows,cols,_ = img.shape
	img = cv2.warpAffine(img,M,(cols+4000,rows+4000))
	lmk = lmk + np.float32([[2000,2000]])
	eye_c = lmk[0]
	mse_c = lmk[1]
	facey = eye_c[1] - (mse_c[1] - eye_c[1])
	facey2 = mse_c[1] + (mse_c[1] - eye_c[1])
	facex = (mse_c[0] + eye_c[0])/2 - (mse_c[1] - eye_c[1])*1.5
	facex2 = (mse_c[0] + eye_c[0])/2 + (mse_c[1] - eye_c[1])*1.5
	total = (mse_c[1] - eye_c[1])*3
	scale = total / 128.0
	print(facex,facex2,facey,facey2)
	faceimg = img[int(facey):int(facey2),int(facex):int(facex2)]
	print(faceimg.shape)
	cv2.imwrite(path,cv2.resize(faceimg,(128,128)))

img = cv2.imread('10.jpg')
lmk = draw_2p(img)
img,lmk = rotate_2p(img, lmk)
get_2p(img,lmk,'10_1.jpg')

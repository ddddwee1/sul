import numpy as np 
import scipy.io as sio 
import network 
# import matplotlib.pyplot as plt 
import model3 as M
import os 
import sulplotter as plotter
import cv2 

joints = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,8], [8,9], [9,10], [1,11], [11,12], [12,13]]
plt = plotter.Plotter2D(usebuffer=False, no_margin=True)

def plot(pts):
	for j in joints:
		xs = [pts[j[0],0], pts[j[1],0]]
		ys = [pts[j[0],1], pts[j[1],1]]
		plt.plot(xs,ys,marker='o', linewidth=5)

def get_prediction(out, scale=8):
	shape = out.shape
	out = out.reshape([shape[0], shape[1]*shape[2], shape[3]])
	scr = np.max(out, axis=1)
	out = np.argmax(out, axis=1)

	res = np.zeros([shape[0], shape[3], 2])
	for i in range(shape[0]):
		for j in range(shape[3]):
			res[i,j,0] = out[i,j] % shape[2]
			res[i,j,1] = out[i,j] // shape[2]
	res = res * scale
	return res, scr

def visualize(x, hmap, num):
	img = (x+0.5) * 255
	img = np.uint8(img)
	img[:,:,:,[2,1,0]] = img[:,:,:,[0,1,2]]

	max_point = get_prediction(hmap.numpy())[0]
	print(max_point.shape)

	for i in range(7):
		if not os.path.exists('./vis/%d/'%i):
			os.mkdir('./vis/%d/'%i)
		if not os.path.exists('./imgs/%d/'%i):
			os.mkdir('./imgs/%d/'%i)
		plt.clear()
		plt.imshow(img[i])
		plot(max_point[i])
		# plt.savefig('./vis/%d/%d.jpg'%(i,num))
		resimg = plt.update(require_img=True)
		cv2.imwrite('./vis/%d/%d.jpg'%(i,num), cv2.cvtColor(resimg, cv2.COLOR_RGB2BGR))
		cv2.imwrite('./imgs/%d/%d.jpg'%(i,num), cv2.cvtColor(img[i], cv2.COLOR_RGB2BGR))


mods = network.ModelBundle()
saver = M.Saver(mods)
saver.restore('./LSTMPM/')

x = sio.loadmat('./frames/1.mat')
cent = np.float32(x['center_map'])
x = np.float32(x['data_1st'])
x = np.transpose(x, [3,1,0,2])
cent = np.transpose(cent, [3,1,0,2])

hmap, h, c = mods.s0(x, x, cent)

visualize(x, hmap, 0)

for i in range(2,152):
	print('Frame:',i)
	x = sio.loadmat('./frames/%d.mat'%i)
	x = np.float32(x['data_nth'])
	x = np.transpose(x, [3,1,0,2])

	hmap, h, c = mods.s1(x, hmap, cent, h, c)
	visualize(x, hmap, i-1)

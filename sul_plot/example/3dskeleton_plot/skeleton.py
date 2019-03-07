import numpy as np 
import pickle 
import plotter
# import cv2 

joints = [[8,9],[9,10], [8,14],[14,15],[15,16], [8,11],[11,12],[12,13], [8,7],[7,0], [0,1],[1,2],[2,3], [0,4],[4,5],[5,6]]

f = open('pts.pkl','rb')
data = pickle.load(f)
f.close()

print(data.shape)

plt = plotter.Plotter3D(usebuffer=True)
plt.show()

for i in range(data.shape[0]):
	pts = data[i]
	pts = pts.reshape([-1,3])

	# plt.clear()
	for j in joints:
		xs = [pts[j[0],2], pts[j[1],2]]
		ys = [pts[j[0],0], pts[j[1],0]]
		zs = [-pts[j[0],1], -pts[j[1],1]]
		lims = [[-2,2], [-2,2], [-2,2]]
		plt.plot(xs,ys,zs,lims=lims, marker='o')
	img = plt.update(require_img=True)
	print(img.shape)

	# print(i)
	# img = plt.get_image()
	# print(img.shape)
	# cv2.imshow('a',img)
	# cv2.waitKey(0)

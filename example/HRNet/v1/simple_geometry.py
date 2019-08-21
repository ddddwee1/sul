import numpy as np 
import cv2 
import scipy.io as sio 
import matplotlib.pyplot as plt 
from scipy.interpolate import griddata
import scipy.stats as st

body_idx = [14,11,4,1]
minor_pairs = [[12,11], [13,12], [14,15], [15,16], [1,2], [2,3], [4,5], [5,6], [1,14],[4,11], [6,18], [3,17]]
minor_points = [15,16, 12,13, 2,3, 5,6, 17,18]
width = [5,5,5,5,5, 5,5,5,5,5, 5,5]

def draw_major(pts, shape):
	res = np.zeros(shape).astype(np.uint8)
	depth = pts[:,2]
	points = pts[:,:2]
	pts = points.astype(int)
	cv2.fillConvexPoly(res, points=pts, color=255)
	gx, gy = np.mgrid[0:shape[0], 0:shape[1]]
	depth_interp = griddata(points, depth, (gy,gx), method='linear', fill_value=0)
	return res, depth_interp

def draw_minor(pt3d, shape, width, color=155):
	
	pt = pt3d[:,:2]
	center = (pt[0] + pt[1]) / 2
	diff = pt[1] - pt[0]
	height = np.sqrt(np.sum(np.square(diff))) / 2.
	# tangent = diff[1] / diff[0]
	# cosine = 1. / np.sqrt(1 + np.square(tangent))
	# sine = tangent * cosine
	hypotenuse = np.sqrt(diff[1]**2 + diff[0]**2) + 1e-5
	cosine = diff[0] / hypotenuse
	sine = diff[1] / hypotenuse

	if width is None:
		width = height

	pts = np.array([[-width, height], [width, height], [width, -height], [-width, -height]])
	rot = np.float32([[-sine, cosine],[cosine, sine]])
	pts = pts.dot(rot.T)
	pts += np.float32([center])

	# fill depth 
	if pt3d[0,1]>pt3d[1,1]:
		d1 = pt3d[0,2]
		d2 = pt3d[1,2]
	else:
		d1 = pt3d[1,2]
		d2 = pt3d[0,2]

	depth = np.float32([[d1],[d1],[d2],[d2]])
	res = np.concatenate([pts, depth], axis=-1)

	res, depth = draw_major(res, shape)
	return res, depth


def test_occlusion(data):
	chn = len(minor_pairs)+1
	canvas = np.zeros([256,256, chn]).astype(np.uint8)
	depth = np.zeros([256,256,chn]).astype(np.float32)
	# compute major (body)
	major_pts = data[np.array(body_idx)]
	major_, major_depth = draw_major(major_pts, (256,256))
	canvas[:,:,0] = major_
	depth[:,:,0] = major_depth
	# compute minor(limbs)
	for i in range(len(minor_pairs)):
		minor_pts = data[np.array(minor_pairs[i])]
		minor_, minor_depth = draw_minor(minor_pts, (256,256), width[i])
		canvas[:,:,i+1] = minor_
		depth[:,:,i+1] = minor_depth
	# visibility score
	visible = np.ones(len(minor_points)).astype(np.float32)
	for idx,pt in enumerate(minor_points):
		for c in range(chn):
			if c>0:
				if pt in minor_pairs[c-1]:
					continue
			x,y,z = int(data[pt,0]), int(data[pt,1]), data[pt,2]
			# print(canvas[y,x,c])
			if canvas[y,x,c]>0 and depth[y,x,c]<z:
				visible[idx] = 0
	mask = np.ones([len(data),3]).astype(np.float32)
	for i,idx in enumerate(minor_points):
		mask[idx] = visible[i]
	return mask 

if __name__=='__main__':
	data = np.float32(sio.loadmat('00000080')['pts'])
	data = data.reshape([-1,3])
	data = data[:-2,:]
	data = (data + 2.5) * 50
	
	mask = test_occlusion(data)
	print(mask)
	
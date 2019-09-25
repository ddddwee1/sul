import pickle 
import numpy as np 
import pickle 

# Joints in H3.6M -- data has 32 joints,
# but only 17 that move; these are the indices.
H36M_NAMES = [''] * 32
H36M_NAMES[0] = 'Hip'
H36M_NAMES[1] = 'RHip'
H36M_NAMES[2] = 'RKnee'
H36M_NAMES[3] = 'RFoot'
H36M_NAMES[6] = 'LHip'
H36M_NAMES[7] = 'LKnee'
H36M_NAMES[8] = 'LFoot'
H36M_NAMES[12] = 'Spine'
H36M_NAMES[13] = 'Thorax'
H36M_NAMES[14] = 'Neck/Nose'
H36M_NAMES[15] = 'Head'
H36M_NAMES[17] = 'LShoulder'
H36M_NAMES[18] = 'LElbow'
H36M_NAMES[19] = 'LWrist'
H36M_NAMES[25] = 'RShoulder'
H36M_NAMES[26] = 'RElbow'
H36M_NAMES[27] = 'RWrist'

def get_17pts(points):
	dim_to_use_x = np.where(np.array([x != '' for x in H36M_NAMES]))[0] * 3
	dim_to_use_y = dim_to_use_x + 1
	dim_to_use_z = dim_to_use_x + 2
	dim_to_use = np.array([dim_to_use_x, dim_to_use_y, dim_to_use_z]).T.flatten()
	points = points[:, dim_to_use]
	return points

def project_point_radial(P, R, T, f, c, k, p):
	"""
	Project points from 3d to 2d using camera parameters
	including radial and tangential distortion
	Args
	P: Nx3 points in world coordinates
	R: 3x3 Camera rotation matrix
	T: 3x1 Camera translation parameters
	f: (scalar) Camera focal length
	c: 2x1 Camera center
	k: 3x1 Camera radial distortion coefficients
	p: 2x1 Camera tangential distortion coefficients
	Returns
	Proj: Nx2 points in pixel space
	D: 1xN depth of each point in camera space
	radial: 1xN radial distortion per point
	tan: 1xN tangential distortion per point
	r2: 1xN squared radius of the projected points before distortion
	"""

	# P is a matrix of 3-dimensional points
	assert len(P.shape) == 2
	assert P.shape[1] == 3

	N = P.shape[0]
	X = R.dot(P.T - T)  # rotate and translate
	XX = X[:2, :] / X[2, :]
	r2 = XX[0, :] ** 2 + XX[1, :] ** 2

	radial = 1 + np.einsum(
		'ij,ij->j', np.tile(k, (1, N)), np.array([r2, r2 ** 2, r2 ** 3]))
	tan = p[0] * XX[1, :] + p[1] * XX[0, :]

	XXX = XX * np.tile(radial + tan, (2, 1)) + \
		np.outer(np.array([p[1], p[0]]).reshape(-1), r2)

	Proj = (f * XXX) + c
	# print(c)
	Proj = Proj.T

	D = X[2,]
	cam_coord = X[:2, :]
	cam_coord = cam_coord.T

	return X.T, Proj

def translate_pts(pts):
	xs = pts[0::2]
	ys = pts[1::2]
	body_center = np.float32([xs[0], ys[0]])

	w = xs.max() - xs.min()
	h = ys.max() - ys.min()
	wh = max(w,h) * 1.4
	corner = body_center - wh / 2
	pts_res = pts.reshape([-1,2]) - np.array([corner[0], corner[1]])
	pts_res = pts_res * 256 / wh 
	scale = 256 / wh 
	return pts_res.tolist(), scale, body_center

def process_batch(pts):
	res = []
	centers = []
	scales = []
	for i in range(len(pts)):
		pp, scale, body_center = translate_pts(pts[i])
		res.append(pp)
		centers.append(body_center)
		scales.append(scale)
	res = np.float32(res)
	scales = np.float32(scales)
	centers = np.float32(centers)
	return res, centers, scales

if __name__=='__main__':

	clip_list = []

	p3d = pickle.load(open('points_3d.pkl','rb'))
	cams = pickle.load(open('cameras_old.pkl','rb'))

	subjects = list(p3d.keys())

	print(subjects)

	for sub in subjects:
		actions = list(p3d[sub].keys())
		for act in actions:
			if sub=='S11' and act=='Directions':
				# corrupted video
				continue
			if sub=='S9':
				continue
			# elif sub=='S9':
			# 	continue
			else:
				for cam in cams[sub].keys():
					buf = {'sub':sub, 'act':act, 'cam':cam}
					clip_list.append(buf)

	data = {}
	data_list = []
	print(len(clip_list))

	maxz = 0
	minz = 99999

	cnt = 0
	for i in clip_list:
		cnt += 1
		sub = i['sub']
		act = i['act']
		cam_name = i['cam']
		# print(cam_name)
		cam = cams[sub][cam_name]

		pts = p3d[sub][act]
		pts = get_17pts(pts)
		pts = pts.reshape([-1, 3])

		pts3d, imgcoord = project_point_radial(pts, **cam)
		pts3d = pts3d.reshape([-1, 17, 3])
		imgcoord = imgcoord.reshape([-1, 17, 2])
		zmax = pts3d[:,:,2].max()
		zmin = pts3d[:,:,2].min()
		# print(zmax)
		# print(zmin)
		if zmax>maxz:
			maxz = zmax
		if zmin<minz:
			minz = zmin
		# pts3d = pts3d - pts3d[:, 0:1, :]
		print(cnt,':',pts3d.shape,imgcoord.shape)
		# print()
		data_list.append([pts3d, imgcoord])
	print('MAX:',maxz, 'MIN:',minz)

with open('points_flatten2.pkl','wb') as f:
	pickle.dump(data_list, f)

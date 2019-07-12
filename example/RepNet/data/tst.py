import pickle 
import numpy as np 
# import util 
import pickle 
from scipy.linalg import orthogonal_procrustes

# Joints in H3.6M -- data has 32 joints,
# but only 17 that move; these are the indices.
H36M_NAMES = [''] * 32
H36M_NAMES[0] = 'Hip' #0
H36M_NAMES[1] = 'RHip' #1
H36M_NAMES[2] = 'RKnee'#2
H36M_NAMES[3] = 'RFoot'#3
H36M_NAMES[6] = 'LHip'#4
H36M_NAMES[7] = 'LKnee'#5
H36M_NAMES[8] = 'LFoot'#6
H36M_NAMES[12] = 'Spine'#7
H36M_NAMES[13] = 'Thorax'#8
H36M_NAMES[14] = 'Neck/Nose'#9
H36M_NAMES[15] = 'Head'#10
H36M_NAMES[17] = 'LShoulder'#11
H36M_NAMES[18] = 'LElbow'#12
H36M_NAMES[19] = 'LWrist'#13
H36M_NAMES[25] = 'RShoulder'#14
H36M_NAMES[26] = 'RElbow'#15
H36M_NAMES[27] = 'RWrist'#16

def get_17pts(points):
	dim_to_use_x = np.where(np.array([x != '' for x in H36M_NAMES]))[0] * 3
	dim_to_use_y = dim_to_use_x + 1
	dim_to_use_z = dim_to_use_x + 2
	dim_to_use = np.array([dim_to_use_x, dim_to_use_y, dim_to_use_z]).T.flatten()
	points = points[:, dim_to_use]
	return points

def procrustes(x, template):
	normx = np.linalg.norm(x)
	normt = np.linalg.norm(template)

	x = x / normx
	t = template / normt
	# print(t)

	R, s = orthogonal_procrustes(t, x)
	# print(np.dot(x, R.T) * s)
	return normx, R, s

# build template matrix
template = np.zeros([4, 3]).astype(np.float32)
template[0] = [1, -1, 0]  #1
template[1] = [-1,-1, 0]  #4
template[2] = [-1, 1, 0]  #11
template[3] = [1, 1, 0]   #14


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
	Proj = Proj.T

	D = X[2,]

	return Proj, D, radial, tan, r2


def align(dt):
	dt = dt - dt[0:1]
	normx, R, s = procrustes(dt[[1, 4, 11, 14]], template)
	dt = dt / normx
	pts_norm = np.dot(dt, R.T) * s 
	return pts_norm

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
			else:
				for cam in cams[sub].keys():
					buf = {'sub':sub, 'act':act, 'cam':cam}
					clip_list.append(buf)

	data_list = []
	print(len(clip_list))

	for i in clip_list:
		print(i)
		sub = i['sub']
		act = i['act']
		cam_name = i['cam']
		cam = cams[sub][cam_name]

		pts = p3d[sub][act]
		pts = get_17pts(pts)
		pts = pts.reshape([-1, 3])

		xyz = cam['R'].dot(pts.T).T

		pts_2d, D = project_point_radial(pts, **cam)[0:2]

		xyz = xyz.reshape([-1, 17, 3])
		pts_2d = pts_2d.reshape([-1, 17, 2])
		
		for j in range(len(xyz)):
			dt = xyz[j]
			dt = align(dt)
			dt2d = pts_2d[j]
			dt2d = dt2d - dt2d[0]
			data_list.append([dt, dt2d])


with open('points_flatten2.pkl','wb') as f:
	pickle.dump(data_list, f)

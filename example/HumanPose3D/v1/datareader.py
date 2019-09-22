import numpy as np 
import model3 as M 
import tensorflow as tf 
import pickle 
import random 

def process_2d(data):
	data_x = data[:,0::3]
	data_y = data[:,1::3]
	data_z = data[:,2::3]
	data_2d = np.stack([data_x, data_y], axis=-1)
	data_2d = data_2d.reshape([-1, 2*17])
	return data_2d, data_z

def generate_mask(x):
	shape = x.shape
	outshape = [x.shape[0], x.shape[1]//3, 1]
	mask = np.random.choice(2, size=outshape, p=[0.2, 0.8]) # edit probability here
	mask = np.concatenate([mask]*2, axis=-1)
	mask = mask.reshape([-1, 34])
	return np.float32(mask) 

def reproj(pts_2d, depth, rot=None):
	if rot is None:
		rot = random.random() * 2 - 1
	rot = rot * np.pi

	pts_2d_x = pts_2d[:,0::2]
	pts_2d_y = pts_2d[:,1::2]
	pts_2d_x_1 = np.cos(rot) * pts_2d_x - np.sin(rot) * depth
	depth_1 = np.sin(rot) * pts_2d_x + np.cos(rot) * depth

	pts_2d = np.stack([pts_2d_x_1, pts_2d_y], axis=-1)
	pts_2d = np.reshape(pts_2d, [-1, 17*2])
	return pts_2d, depth_1

def pad_head_or_tail(data, size, missing_head=True):
	size = size * 2 + 1 
	if len(data) < size:
		if missing_head:
			append_head = np.stack([data[0]]*(size-len(data)),axis=0)
			res = np.concatenate([append_head, data], axis=0)
		else:
			append_tail = np.stack([data[-1]]*(size-len(data)),axis=0)
			res = np.concatenate([data, append_tail], axis=0)
	elif len(data)==size:
		res = np.float32(data)
	else:
		raise ValueError('data size is larger than specified temporal elapse')
	return res 

def process_pts(x):
	x_2d, x_depth = process_2d(x)
	x_2d, x_depth = reproj(x_2d, x_depth)
	mask = generate_mask(x)
	# maybe do reproject and re-normalization
	return x_2d, x_depth, mask

SAMPLE_INTERVALS = [2,3,5,10]
class DataReader(M.ThreadReader):
	def _get_data(self):
		res = []
		data = pickle.load(open('points_flatten.pkl', 'rb'))
		for i in data:
			for SAMPLE_INTERVAL in SAMPLE_INTERVALS:
				for s in range(SAMPLE_INTERVAL):
					res.append(i[s::SAMPLE_INTERVAL])
		return res 

	def _next_iter(self):
		res = []
		batch = random.sample(self.data, self.bsize)
		for b in batch:
			pos = random.randint(0, len(b)-1)
			sample = b[max(0,pos-self.temp) : min(len(b),pos+self.temp+1)]
			sample = pad_head_or_tail(sample, self.temp, pos-self.temp<0)
			res.append(sample)
		res = np.float32(res)
		# print(res.shape)
		return res 

	def _process_data(self, x):
		return process_pts(x)

	def _post_process(self, x):
		x = list(zip(*x))
		x = [np.float32(_) for _ in x]
		return x 

if __name__=='__main__':
	reader = DataReader(16, temp=32)
	data = reader.get_next()
	print(len(data))
	print(data[0].shape)
	print(data[1].shape)
	print(data[2].shape)

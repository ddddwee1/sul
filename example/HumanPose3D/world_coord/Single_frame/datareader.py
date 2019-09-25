import numpy as np 
import random
import cv2 
import scipy.stats as st
import pickle 
import model3 as M 

def translate_pts(pts):
	xs = pts[:,0]
	ys = pts[:,1]
	body_center = np.float32([xs[0], ys[0]])

	w = xs.max() - xs.min()
	h = ys.max() - ys.min()
	wh = max(w,h) * 1.4
	corner = body_center - wh / 2
	pts_res = pts.reshape([-1,2]) - body_center
	pts_res = pts_res * 256 / wh 
	scale = 256 / wh 
	return pts_res, scale

class data_reader(M.ThreadReader):
	def _get_data(self):
		# SAMPLE_INTERVALS = [2, 3, 5, 10]
		SAMPLE_INTERVALS = [3]
		dataraw = pickle.load(open('points_flatten2.pkl', 'rb'))
		data = []
		for i in dataraw:
			for SAMPLE_INTERVAL in SAMPLE_INTERVALS:
				for s in range(SAMPLE_INTERVAL):
					data.append([i[0][s::SAMPLE_INTERVAL], i[1][s::SAMPLE_INTERVAL]])
		return data

	def _process_sample(self, data, size, missing_head=True):
		# print(len(data))
		size = size*2+1
		if len(data)<size:
			if missing_head:
				append_head = np.stack([data[0]]*(size-len(data)),axis=0)
				res = np.concatenate([append_head, data], axis=0)
			else:
				append_tail = np.stack([data[-1]]*(size-len(data)),axis=0)
				res = np.concatenate([data, append_tail], axis=0)
		elif len(data)==size:
			res = data 
		else:
			raise ValueError('data size is larger than specified temporal elapse')
		return res 

	def _next_iter(self):
		def process(data, sample_point):
			data = data[ max(0,point-self.temp) : min(len(data), point+self.temp+1) ]
			data = self._process_sample(data, self.temp, point-self.temp<=0)
			return data 
		res = []
		batch = random.sample(self.data, self.bsize)
		for sample3d, sample2d in batch:
			point = random.randint(0, len(sample3d)-1)
			sample3d = process(sample3d, point)
			sample2d = process(sample2d, point)
			res.append([sample3d,sample2d])
		return res 

	def _process_data(self, item):
		pts3d = item[0]
		pts2d = item[1]

		# print(pts2d.shape)
		body_center = pts2d[:,0:1, :]

		buff = []
		scales = []
		for p2d in pts2d:
			p2d, scale = translate_pts(p2d)
			buff.append(p2d)
			scales.append(scale)
		buff = np.float32(buff)
		scales = np.float32(scales)
		z_value = pts3d[:,:,2:3]

		return body_center, scales, pts2d, z_value

	def _post_process(self, x):
		x = list(zip(*x))
		x = [np.float32(_) for _ in x]
		return x 

	def concat_all(self, xs):
		shape = xs[0].shape
		res = [i.reshape([shape[0], shape[1], -1]) for i in xs]
		res = np.concatenate(res, axis=-1)
		return res 

if __name__=='__main__':
	reader = data_reader(1, temp=32)
	reader.get_next()
	# hmap, res3d = reader.get_next()
	# print(hmap.shape)
	# print(res3d[0,0])
	# hmap = np.amax(hmap[0,0], axis=-1)
	# hmap = np.uint8(hmap)
	# cv2.imshow('a', hmap)
	# cv2.waitKey(0)

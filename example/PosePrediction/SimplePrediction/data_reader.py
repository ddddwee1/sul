import numpy as np 
import pickle 
import random

SAMPLE_INTERVALS = [2,3,5]

class data_reader():
	def __init__(self):
		data = pickle.load(open('points_flatten.pkl','rb'))
		# data2 = pickle.load(open('mpii.pkl','rb'))
		# data = data + data2
		# print(data[0].shape)
		print(type(data))

		self.data = []
		for i in data:
			for SAMPLE_INTERVAL in SAMPLE_INTERVALS:
				for s in range(SAMPLE_INTERVAL):
					self.data.append(i[s::SAMPLE_INTERVAL])

	def process_sample(self, data, size, missing_head=True):
		size = size*2+1
		if len(data)<size:
			if missing_head:
				append_head = np.stack([data[0]]*(size-len(data)),axis=0)
				# print(data.shape)
				# print(append_head.shape)
				res = np.concatenate([append_head, data], axis=0)
			else:
				append_tail = np.stack([data[-1]]*(size-len(data)),axis=0)
				# print(data.shape)
				# print(append_head.shape)
				res = np.concatenate([data, append_tail], axis=0)
		elif len(data)==size:
			res = data 
		else:
			raise ValueError('data size is larger than specified temporal elapse')

		return res 

	def get_next(self, bsize, size):
		res = []
		for i in range(bsize):
			sample = random.choice(self.data)
			point = random.randint(0,len(sample)-1)
			sample = sample[ max(0,point-size) : min(len(sample)-1, point+size+1) ]
			if point-size<0:
				missing_head = True
			else:
				missing_head = False
			sample = self.process_sample(sample, size, missing_head)

			res.append(sample)
		res = np.float32(res)
		res = res.transpose((1,0,2))
		return res



if __name__=='__main__':
	r = data_reader()

	data_batch, label_batch = r.get_sample(0,64)

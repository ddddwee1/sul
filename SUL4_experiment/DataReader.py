from multiprocessing import Pool 
import numpy as np 
import random 

# A simple data reader.
# You must set: process function, sample policy (default random sample), process number (default 1), gpu number (default 1)

class DataReader():
	def __init__(self, bsize, sample_policy='RANDOM', processes=1, gpus=1):
		# assert sample_policy in ['RANDOM', 'EPOCH', 'CUSTOM'], 'Policy must be one of "RANDOM" / "EPOCH" / "CUSTOM"'
		assert sample_policy in ['RANDOM', 'EPOCH'], 'Policy must be one of "RANDOM" / "EPOCH"'
		assert processes >= 1, 'Processes must be a positive integer'
		assert gpus >= 1,'GPU number must be a positive integer'
		self.data = None
		self.process_fn = lambda x:x
		self.post_process_fn = None
		self.maxpos = None
		self.bsize = bsize
		self.gpus = gpus
		self.policy = sample_policy
		self.pool = Pool(processes=processes)

	def set_data(self, data):
		self.data = data 
		if self.policy == 'EPOCH':
			self.idx = list(range(len(self.data)))
			self.position = 0
			self.epoch = 0
		print('DataReader: Data set.')

	def set_process_fn(self, fn):
		self.process_fn = fn 
		print('DataReader: Process function set.')

	def set_post_process_fn(self, fn):
		self.post_process_fn = fn 

	def get_next(self):
		assert self.ps is not None, 'You must call prefetch before the first iteration'
		result = self.ps.get()
		if self.gpus > 1:
			result = self.split_data(result)
			result = [self.post_process_fn(i) for i in result]
		else:
			result = self.post_process_fn(result)
		self.prefetch()
		return result

	def prefetch(self):
		assert self.process_fn is not None, 'You must set process function first'
		if self.policy == 'RANDOM':
			idx = random.sample(self.idx, self.bsize)
		if self.policy == 'EPOCH':
			if self.position+self.bsize>len(self.idx):
				random.shuffle(self.idx)
				self.position = 0
				self.epoch += 1
			idx = self.idx[self.position:self.position+self.bsize]

		res = [self.data[i] for i in idx]
		self.ps = self.pool.map_async(self.process_fn, res)

	def split_data(self, data):
		res = []
		length = len(data)
		len_split = length//self.gpus + int(length%self.gpus>0)
		for i in range(self.gpus):
			res.append(data[len_split * i : min(len(data) , len_split * i + len_split)])
		return res 

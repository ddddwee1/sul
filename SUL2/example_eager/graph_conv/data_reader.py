import numpy as np 
import pickle 
import os 

class data_reader():
	def __init__(self):

		if os.path.exists('./data.pkl'):
			print('Loading from pickle')
			data = pickle.load(open('./data.pkl','rb'))
			self.adj, self.paper_category, self.features = data
			print('Loaded')

		else:
			print('No pkl found, generating data...')
			num_id = 0

			explored_categories = {}

			paper_id = {}
			paper_category = {}
			features = []
			f = open('./cora/cora.content')
			for i in f:
				i = i.strip()
				i = i.split('\t')
				pprid = int(i[0])
				category = i[-1]
				feature = i[1:-1]
				feature = [float(item) for item in feature]
				feature = np.float32(feature)
				features.append(feature)
				paper_id[pprid] = num_id

				if not category in explored_categories:
					num = len(explored_categories)
					print(category)
					explored_categories[category] = num 

				paper_category[num_id] = explored_categories[category]

				num_id+=1

			citation = []
			f = open('./cora/cora.cites')
			for i in f:
				i = i.strip()
				i = i.split('\t')
				cited_ppr = int(i[0])
				cited_id = paper_id[cited_ppr]
				citing_ppr = int(i[1])
				citing_id = paper_id[citing_ppr]

				citation.append([cited_id, citing_id])

			self.max_id = num_id
			self.paper_category = paper_category
			self.citation = citation

			self.adj = self.get_adj_mtx()
			self.features = np.float32(features)

			data = [self.adj, self.paper_category, self.features]

			pickle.dump( data, open('./data.pkl','wb'))
			print('Dump finished')

	def get_adj_mtx(self):
		mtx = np.zeros([self.max_id, self.max_id], np.float32)
		for i in self.citation:
			mtx[i[0], i[1]] = 1
			mtx[i[1], i[0]] = 1
		return mtx 

	def process_data(self, one_hot=True):
		import random
		random.seed(2019)
		indices = list(range(2708))
		self.indices = random.sample(indices, 140)
		self.labels = [self.paper_category[i] for i in self.indices]
		if one_hot:
			self.labels = [np.eye(7)[i] for i in self.labels]
		self.indices = [[i] for i in self.indices]

	def get_data(self):
		category = [self.paper_category[i] for i in range(2708)]
		return self.features, self.adj, self.indices, self.labels, category

if __name__=='__main__':
	dt = data_reader()
	print(dt.adj.shape)
	cate = dt.paper_category
	a = {}
	for k in cate:
		a[cate[k]] = 1
	print(len(a))

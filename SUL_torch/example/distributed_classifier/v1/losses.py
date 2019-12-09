import torch
import torch.nn as nn 
import torch.nn.init as init 
import torch.nn.functional as F 
import Model as M 
import numpy as np 
from torch.nn.parameter import Parameter
from torch.nn.parallel import replicate, scatter, parallel_apply, gather
import concurrent.futures

class MarginalCosineLayer(M.Model):
	def initialize(self, num_classes):
		self.classifier = M.Dense(num_classes, usebias=False, norm=True)
	def forward(self, x, label, m1=1.0, m2=0.0, m3=0.0):
		x = self.classifier(x)
		if not (m1==1.0 and m2==0.0 and m3==0.0):
			# print('MarginalCosineLayer used.')
			idx = torch.arange(0,label.size(0), dtype=torch.long)
			t = x[idx, label]
			t = torch.acos(t)
			if m1!=1.0:
				t = t * m1 
			if m2!=0.0:
				t = t + m2 
			t = torch.cos(t)
			x[idx, label] = t - m3 
		return x 

def classify(feat, weight, **kwargs):
	feat = feat / feat.norm(p=2, dim=1, keepdim=True)
	weight = weight / weight.norm(p=2, dim=1, keepdim=True)
	x = F.linear(feat, weight, None)
	return x 

def arccos(x, label, m1=1.0, m2=0.0, m3=0.0):
	if not (m1==1.0 and m2==0.0 and m3==0.0):
		idx = torch.where(label>=0)[0]
		t = x[idx, label]
		t = torch.acos(t)
		if m1!=1.0:
			t = t * m1 
		if m2!=0.0:
			t = t + m2 
		t = torch.cos(t)
		x[idx, label] = t - m3 
	return x

class MarginalCosineLayerScatter(M.Model):
	def initialize(self, num_classes, device_ids):
		self.outsize = self.num_classes = num_classes
		self.device_ids = device_ids

	def build(self, *inputs):
		print('Building MarginalCosineLayer...')
		self.insize = inputs[0].shape[1]
		self.weights = []
		split = self.num_classes // len(self.device_ids)
		outs = [split*i for i in range(len(self.device_ids))]
		outs.append(self.num_classes)
		self.outs = outs 
		# self.weight = Parameter(torch.Tensor(self.outsize, self.insize))

		for i in range(len(outs)-1):
			weight = Parameter(torch.Tensor(outs[i+1]-outs[i], self.insize).to(self.device_ids[i]))
			# weight = weight.to(self.device_ids[i])
			self.register_parameter('weight%d'%i, weight)
			init.normal_(weight, std=0.01)
			
			print(weight.device)
			# print(weight)
			self.weights.append(weight)

	def forward(self, feat, labels, **kwargs):
		# weight, feature and labels 
		weight_scattered = self.weights
		feat_copies = [feat.to(i) for i in self.device_ids]
		input_scattered = list(zip(feat_copies, weight_scattered))
		# kwargs 
		kwargs_scattered = scatter(kwargs, self.device_ids)
		# apply parallel 
		moduels = [classify]*len(self.device_ids)
		# print(len(moduels), len(input_scattered))
		results_scattered = parallel_apply(moduels, input_scattered, kwargs_scattered, self.device_ids)
		# print(len(results_scattered[0]))
		# gather results 
		results = gather(results_scattered, 0, dim=1)
		results = arccos(results, labels, **kwargs)
		return results

	def save(self):
		import pickle 
		res = []
		for w in self.weights:
			res.append(w.item())
		res = np.concatenate(res,axis=1)
		return res 

class NLLLoss(M.Model):
	def initialize(self):
		self.lgsm = nn.LogSoftmax(dim=1)
		# self.lsfn = nn.NLLLoss()
	def forward(self, x, label):
		x = self.lgsm(x)
		label = label.unsqueeze(-1)
		loss = torch.gather(x, 1, label)
		loss = -loss.mean()
		return loss

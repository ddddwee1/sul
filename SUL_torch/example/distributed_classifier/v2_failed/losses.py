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

def classify(feat, weight, label, m1=1.0, m2=0.5, m3=0.0, s=64):
	feat = feat / feat.norm(p=2, dim=1, keepdim=True)
	weight = weight / weight.norm(p=2, dim=1, keepdim=True)
	x = F.linear(feat, weight, None)
	bsize = feat.shape[0]
	if not (m1==1.0 and m2==0.0 and m3==0.0):
		idlen = weight.shape[0]
		idx = torch.where((label>=0) & (label<idlen))[0]
		if idx.shape[0]==0:
			# print('Not in this patch.')
			x = x * s
			xexp = torch.exp(x)
			xsum = xexp.sum(dim=1, keepdim=True)
			return x, xexp, xsum
		label = label[(label>=0) & (label<idlen)]

		t = x[idx, label]
		t = torch.acos(t)
		if m1!=1.0:
			t = t * m1 
		if m2!=0.0:
			t = t + m2 
		t = torch.cos(t)
		x[idx, label] = t - m3 

	x = x * s 
	xexp = torch.exp(x)
	xsum = xexp.sum(dim=1, keepdim=True)
	return x, xsum, xexp

class FakeTensor():
	def __init__(self, x, ctx):
		self.data = x 
		self.ctx = ctx 

	def backward(self):
		self.ctx.backward()

	def __float__(self):
		return float(self.data)

class NLLScatter():
	def __init__(self):
		self.exe = concurrent.futures.ThreadPoolExecutor(max_workers=8)
	def __call__(self, inp, labels):
		logits = [i[0] for i in inp]
		maxs = [[i[1]] for i in inp]
		exps = [i[2] for i in inp]
		with torch.no_grad():
			maxttl = gather(maxs, 0, dim=1)[0]
			maxttl = maxttl.sum(dim=1,keepdim=True)
			logits_sm = []
			grads = []
			for e,label in zip(exps, labels):
				dev = e.device
				e = e / maxttl.to(dev)
				# modify gradient
				idx = torch.where(label>=0)[0]
				if idx.shape[0]!=0:
					label = label[idx]
					# print(e.shape)
					# print(idx.shape)
					# print(label.shape)
					logits_sm.append(e[idx,label].cpu().detach().numpy())
					e[idx,label] -= 1.
				grads.append(e)

		result = np.concatenate(logits_sm)
		result = - np.log(result)
		result = result.mean()
		self.grads = grads
		self.logits = logits
		return FakeTensor(result, self)

	def backward(self):
		devices = len(self.logits)
		futures = []
		for i in range(devices):
			######### BUGS HERE ##########
			######### BUGS HERE ##########
			######### BUGS HERE ##########
			######### BUGS HERE ##########
			# future = self.exe.submit(self.logits[i].backward, gradient=self.grads[i]/devices, retain_graph= i!=(devices-1))
			future = self.exe.submit(self.logits[i].backward, gradient=self.grads[i]/devices, retain_graph=True) 
			# self.logits[i].backward(gradient=self.grads[i] / devices, retain_graph=True)
			######### BUGS HERE ##########
			######### BUGS HERE ##########
			######### BUGS HERE ##########
			######### BUGS HERE ##########
			######### BUGS HERE ##########
			futures.append(future)
		# free mem
		for f in futures:
			f.result()
		del self.logits
		del self.grads

NLLLoss = NLLScatter()

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
		# for w in weight_scattered:
			# print(w.shape)
		feat_copies = [feat.to(i) for i in self.device_ids]
		# labels_scattered = [labels-i for i in self.outs[:-1]]
		labels_scattered = []
		for i in range(len(self.outs)-1):
			labelsnew = labels.clone()
			labelsnew[(labelsnew>=self.outs[i+1]) | (labelsnew<self.outs[i])] = -1
			labelsnew = labelsnew - self.outs[i]
			labels_scattered.append(labelsnew)
		input_scattered = list(zip(feat_copies, weight_scattered, labels_scattered))
		# kwargs 
		kwargs_scattered = scatter(kwargs, self.device_ids)
		# apply parallel 
		moduels = [classify]*len(self.device_ids)
		# print(len(moduels), len(input_scattered))
		results_scattered = parallel_apply(moduels, input_scattered, kwargs_scattered, self.device_ids)
		# print(len(results_scattered[0]))
		# gather results 
		# results = gather(results_scattered, 0, dim=1)
		# results = arccos(results, labels, **kwargs)

		results = NLLLoss(results_scattered, labels_scattered)
		return results

	def save(self):
		import pickle 
		res = []
		for w in self.weights:
			res.append(w.item())
		res = np.concatenate(res,axis=1)
		return res 



# class NLLLoss(M.Model):
# 	def initialize(self):
# 		self.lgsm = nn.LogSoftmax(dim=1)
# 		# self.lsfn = nn.NLLLoss()
# 	def forward(self, x, label):
# 		x = self.lgsm(x)
# 		label = label.unsqueeze(-1)
# 		loss = torch.gather(x, 1, label)
# 		loss = -loss.mean()
# 		return loss

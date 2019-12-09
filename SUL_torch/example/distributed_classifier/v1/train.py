import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
import Model as M 
import losses 
import resnet 
import util 
import datareader 
import time 

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def initialize(batch, model):
	model.eval()
	batch = [i[0:1] if isinstance(i,torch.Tensor) else torch.from_numpy(i[0:1]) for i in batch]
	res = model(*batch)
	return res 

if __name__=='__main__':
	# initialize datareader 
	BSIZE = 128 * 6
	reader = datareader.get_datareader('../dataset/emore_asia_outs.txt', BSIZE, processes=16)

	# initialize model 
	gpus = (0,1,2,3)
	model = resnet.Res100()
	arcface = losses.MarginalCosineLayerScatter(num_classes = reader.max_label+1, device_ids=gpus)
	batch = reader.get_next()
	_ = initialize([batch[0]], model)
	y = initialize( [_, batch[1]] , arcface)
	# print(float(y))
	# y.backward()

	# map to gpus
	model = nn.DataParallel(model, device_ids=gpus).cuda()
	saver = M.Saver(model)
	saver.restore('./model/')
	saver_classifier = M.Saver(arcface)
	saver_classifier.restore('./classifier/')

	# optimizer
	optim = torch.optim.SGD([{'params':model.parameters()}, {'params':arcface.parameters()}], lr=0.1, momentum=0.9, weight_decay=0.0005)
	model.train()
	arcface.train()

	# loss function 
	MAXEPOCH = 20
	lossfunc = losses.NLLLoss()
	t00 = time.time()
	for e in range(MAXEPOCH):
		for i in range(reader.iter_per_epoch):
			# t0 = time.time()
			batch = reader.get_next()
			# t1 = time.time()
			# print('Datatime:',t1-t0)
			imgs = torch.from_numpy(batch[0])
			labels = torch.from_numpy(batch[1])
			labels = labels.cuda()
			# t2 = time.time()
			# print('Convert time:',t2-t1)

			# go training 
			optim.zero_grad()
			feats = model(imgs)
			logits = arcface(feats, labels, m1=1.0, m2=0.5, m3=0.0) 
			# t3 = time.time()
			# print('forward time:', t3-t2)
			loss = lossfunc(logits*64, labels)
			loss2 = loss / len(gpus)
			# t4 =time.time()
			# print('Loss time:', t4-t3)
			loss2.backward()
			optim.step()
			# t5 = time.time()
			# print('Backward time:', t5-t4)

			lr = optim.param_groups[0]['lr']
			if i%10==0:
				t11 =time.time()
				speed = BSIZE * 10 / (t11-t00)
				t00 = t11
				print('Epoch:%03d\tIter:%06d/%06d\tLoss:%.4f\tLR:%.1e\tSpeed:%.2f'%(e,i,reader.iter_per_epoch,loss, lr, speed))
			if i%2000==0 and i>0:
				saver.save('./model/%04d_%06d.pth'%(e,i))
		if e%2==1:
			newlr = lr * 0.1 
			for param_group in optim.param_groups:
				param_group['lr'] = newlr
		saver.save('./model/%04d_%06d.pth'%(e,i))
		saver_classifier.save('./saver_classifier/%04d_%06d.pth'%(e,i))

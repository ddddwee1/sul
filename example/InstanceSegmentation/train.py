import tensorflow as tf 
import model3 as M 
import numpy as np 
import data_reader
import network
import config
import cv2 

class Detector(M.Model):
	def initialize(self):
		out_channel = len(config.anchor_shape) * len(config.anchor_scale) * ( 5 + config.categories + config.seg_size)
		self.backbone = network.ResNet()
		self.FPN = network.FeaturePyramid()
		self.prediction = M.ConvLayer(1, out_channel)
	def forward(self, x):
		x = self.backbone(x)
		x = self.FPN(*x)
		x = self.prediction(x)
		return x 

def conf_loss(fmap, label, mask):
	weight = tf.stop_gradient(tf.square(tf.sigmoid(fmap) - label))
	loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=fmap, labels=label) * weight) / (tf.reduce_sum(weight) + 1e-5)
	return loss 

def coord_loss(fmap, label, mask):
	loss = tf.reduce_sum(tf.square( fmap - label ) * mask) / (tf.reduce_sum(mask) + 1e-5)
	return loss 

def cls_loss(fmap, label, mask):
	# weight = tf.stop_gradient(tf.abs(tf.sigmoid(fmap) - label) * mask)
	weight = mask 
	idx = tf.where(mask==1)
	fmap = tf.gather_nd(fmap, idx)
	label = tf.gather_nd(label, idx)
	mask = tf.gather_nd(mask, idx)
	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fmap, labels=label) )
	return loss 

def seg_loss(fmap, label, mask):
	weight = mask
	idx = tf.where(mask==1)
	fmap = tf.gather_nd(fmap, idx)
	label = tf.gather_nd(label, idx)
	mask = tf.gather_nd(mask, idx)
	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fmap, labels=label) )
	return loss 
	
# build whatever loss function you like
def gradient_loss(x, model):
	data, label, mask = x
	out_channel = len(config.anchor_shape) * len(config.anchor_scale)
	with tf.GradientTape() as tape:
		# print(1)
		output = model(data)
		confls = conf_loss(output[:,:,:,:out_channel], label[:,:,:,:out_channel], mask[:,:,:,:out_channel])
		coordls = coord_loss(output[:,:,:,out_channel:out_channel*5], label[:,:,:,out_channel:out_channel*5], mask[:,:,:,out_channel:out_channel*5])
		# print(2)
		clsls = cls_loss(output[:,:,:,out_channel*5:out_channel*(5+config.categories)], label[:,:,:,out_channel*5:out_channel*(5+config.categories)], mask[:,:,:,out_channel*5:out_channel*(5+config.categories)])
		# print(3)
		maskls = seg_loss(output[:,:,:,out_channel*(5+config.categories):], label[:,:,:,out_channel*(5+config.categories):], mask[:,:,:,out_channel*(5+config.categories):])
		# print(4)
		# wd = 0.0005
		# w_reg = wd * 0.5 * sum([tf.reduce_sum(tf.square(w)) for w in model.trainable_variables]) 
		loss_overall = confls * 10.0 + coordls * 5.0 + clsls * 1.0 + maskls * 5.0 
	grads = tape.gradient(loss_overall, model.trainable_variables)
	# print(5)
	return grads, [confls, coordls, clsls, maskls]

tf.keras.backend.set_learning_phase(False)
reader = data_reader.DataReader(4)

'''
###### parallel training ######
with tf.device('/cpu:0'):
	net = Detector()
	saver = M.Saver(net)
	saver.restore('./model/')
	opt = tf.optimizers.Adam(0.0001)
	_ = net(np.ones([1,512,512,3]).astype(np.float32)) # initialize step

	pt = M.ParallelTraining(net, opt, devices=[0,1,2,3], grad_loss_fn=gradient_loss)
	for it in range(config.maxiter+1):
		batch = reader.get_next()
		batch_distribute = pt.split_data(batch) # split data for multiple GPUs. Need to run explicitly
		losses = pt.train_step(batch_distribute) 

		if it%10==0:
			confls = tf.reduce_mean([_[0] for _ in losses])
			coordls = tf.reduce_mean([_[1] for _ in losses])
			print('ITER:%d\tConfLoss:%.4f\tCoordLoss:%.4f'%(it, confls, coordls))

		if it%config.save_interval==0 and it>0:
			saver.save('./model/%d.ckpt'%it)

'''

##### single GPU training #####
net = Detector()
saver = M.Saver(net.backbone)
saver.restore('./pretrained_model/')
opt = tf.optimizers.Adam(0.0001)
saver = M.Saver(net, opt)
saver.restore('./model/')
for it in range(config.maxiter+1):
	batch = reader.get_next()
	grad, losses = gradient_loss(batch, net)
	opt.apply_gradients(zip(grad, net.trainable_variables))
	if it%10==0:
		confls, coordls, clsls, maskls = losses
		print('ITER:%d\tConfLoss:%.4f\tCoordLoss:%.4f\tClassLoss:%.4f\tMaskLoss:%.4f'%(it, confls, coordls, clsls, maskls))
		output = net(batch[0]).numpy()
		img_pred, seg_pred = data_reader.visualize(batch[0][0], output[0], sig=True)
		img_gt, seg_gt = data_reader.visualize(batch[0][0], batch[1][0])
		cv2.imshow('imggt', img_gt)
		cv2.imshow('imgpred', img_pred)
		cv2.imshow('seggt', seg_gt)
		cv2.imshow('segpred', seg_pred)
		cv2.waitKey(5)
	if it%config.save_interval==0 and it>0:
		saver.save('./model/%d.ckpt'%it)

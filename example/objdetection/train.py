import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import tensorflow as tf 
import model3 as M 
import numpy as np 
import data_reader
import network
import config
import cv2 

class Detector(M.Model):
	def initialize(self):
		out_channel = len(config.anchor_shape) * len(config.anchor_scale) * ( 5 + config.categories)
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
	loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=fmap, labels=label) * weight) / (tf.reduce_sum(weight) + 1e-5)
	return loss 

# build whatever loss function you like
def gradient_loss(x, model):
	data, label, mask = x
	out_channel = len(config.anchor_shape) * len(config.anchor_scale)
	with tf.GradientTape() as tape:
		output = model(data)
		confls = conf_loss(output[:,:,:,:out_channel], label[:,:,:,:out_channel], mask[:,:,:,:out_channel])
		coordls = coord_loss(output[:,:,:,out_channel:out_channel*5], label[:,:,:,out_channel:out_channel*5], mask[:,:,:,out_channel:out_channel*5])
		clsls = cls_loss(output[:,:,:,out_channel*5:], label[:,:,:,out_channel*5:], mask[:,:,:,out_channel*5:])
		wd = 0.0005
		w_reg = wd * 0.5 * sum([tf.reduce_sum(tf.square(w)) for w in model.trainable_variables]) 
		loss_overall = confls * 10.0 + coordls * 5.0 + clsls * 1.0 + w_reg
	grads = tape.gradient(loss_overall, model.trainable_variables)
	return grads, [confls, coordls, clsls]

tf.keras.backend.set_learning_phase(False)
reader = data_reader.DataReader(8)


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
		confls, coordls, clsls = losses
		print('ITER:%d\tConfLoss:%.4f\tCoordLoss:%.4f\tClassLoss:%.4f'%(it, confls, coordls, clsls))
		# output = net(batch[0]).numpy()
		# img_pred = data_reader.visualize(batch[0][0], output[0])
		# img_gt = data_reader.visualize(batch[0][0], batch[1][0])
		# cv2.imshow('imggt', img_gt)
		# cv2.imshow('imgpred', img_pred)
		# cv2.waitKey(5)
	if it%config.save_interval==0 and it>0:
		saver.save('./model/%d.ckpt'%it)

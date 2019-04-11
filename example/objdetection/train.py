import tensorflow as tf 
import model3 as M 
import numpy as np 
import data_reader
import network
import config

class Detector(M.Model):
	def initialize(self):
		out_channel = len(config.anchor_shape) * len(config.anchor_scale)
		self.backbone = network.ResNet()
		self.FPN = network.FeaturePyramid()
		self.prediction = M.ConvLayer(1, out_channel*config.map_per_channel)

	def forward(self, x):
		x = self.backbone(x)
		x = self.FPN(*x)
		x = self.prediction(x)
		return x 

def conf_loss(fmap, label, mask):
	weight = tf.stop_gradient((tf.sigmoid(fmap) - label)*mask)
	loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=fmap, labels=label) * weight) / tf.reduce_sum(weight)
	return loss 

def coord_loss(fmap, label, mask):
	loss = tf.reduce_mean(tf.square( fmap - label ) * mask)
	return loss 

def att_loss(fmap, label, mask):
	# used for future
	weight = tf.stop_gradient((tf.sigmoid(fmap) - label)*mask)
	loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=fmap, labels=label) * weight) / tf.reduce_sum(weight)
	return loss 

# build whatever loss function you like
def gradient_loss(x, model):
	data, label, mask = x
	out_channel = len(config.anchor_shape) * len(config.anchor_scale)
	with tf.GradientTape() as tape:
		output = model(x)
		confls = conf_loss(output[:,:,:,:out_channel], label[:,:,:,:out_channel])
		coordls = coord_loss(output[:,:,:,out_channel:out_channel*5], label[:,:,:,out_channel:out_channel*5])

		loss_overall = confls * 2.0 + coord_loss * 0.5
	grads = tape.gradient(loss_overall, model.trainable_variables)
	return loss_overall, [confls, coordls]


reader = data_reader.DataReader()

'''
###### parallel training ######
with tf.device('/cpu:0'):
	net = Detector()
	saver = M.Saver(net)
	saver.load('./model/')
	opt = tf.optimizers.Adam(0.0001)
	_ = net(np.ones([1,512,512,3]).astype(np.float32)) # initialize step

	pt = M.ParallelTraining(net, opt, devices=[0,1,2,3], grad_loss_fn=gradient_loss)
	for it in range(config.maxiter):
		batch = reader.get_next()
		batch_distribute = pt.split_data(batch) # split data for multiple GPUs. Need to run explicitly
		losses = pt.train_step(batch_distribute) 

		if it%10==0:
			confls = tf.reduce_mean([_[0] for _ in losses])
			coordls = tf.reduce_mean([_[1] for _ in losses])
			print('ITER:%d\tConfLoss:%.4f\tCoordLoss:%.4f'%(it, confls, coordls))

		if it%5000==0 and it>0:
			saver.save('./model/%d.ckpt'%it)

'''

##### single GPU training #####
net = Detector()
saver = M.Saver(net)
saver.load('./model/')
opt = tf.optimizers.Adam(0.0001)
for it in range(config.maxiter):
	batch = reader.get_next()
	grad, losses = gradient_loss(batch, net)
	opt.apply_gradients(zip(grad, net.trainable_variables))
	if it%10==0:
		confls = tf.reduce_mean([_[0] for _ in losses])
		coordls = tf.reduce_mean([_[1] for _ in losses])
		print('ITER:%d\tConfLoss:%.4f\tCoordLoss:%.4f'%(it, confls, coordls))
	if it%5000==0 and it>0:
		saver.save('./model/%d.ckpt'%it)

import model3 as M 
import network
import tensorflow as tf 
import data_reader
import numpy as np 
import config
import visualize
import cv2 

class Model(M.Model):
	def initialize(self):
		out_channel = config.categories
		self.backbone = network.ResNet()
		self.FPN = network.FeaturePyramid()
		self.prediction = M.ConvLayer(1, out_channel)
		self.upsample = M.BilinearUpSample(8)
	def forward(self, x):
		x = self.backbone(x)
		x = self.FPN(*x)
		x = self.prediction(x)
		x = self.upsample(x)
		return x 

def grad_loss(x, model):
	data, lab = x
	with tf.GradientTape() as tape:
		out = model(data)
		weight = tf.stop_gradient( tf.reduce_mean(tf.abs(lab - tf.nn.softmax(out, -1) ), -1) )
		ls = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=lab, logits=out) *weight) / tf.reduce_sum(weight)
	grad = tape.gradient(ls, model.trainable_variables)
	return grad, [ls, out]

tf.keras.backend.set_learning_phase(False)
reader = data_reader.DataReader(config.batch_size)

model = Model()
optim = tf.optimizers.Adam(0.0001)
saver = M.Saver(model.backbone)
saver.restore('./pretrained_model/')
saver = M.Saver(model, optim)
saver.restore('./model/')

for it in range(config.max_iter + 1):
	batch = reader.get_next()
	grad, losses = grad_loss(batch, model)
	optim.apply_gradients(zip(grad, model.trainable_variables))
	if it%10==0:
		ls = losses[0]
		out = losses[1]
		print('ITER:%d\tLoss:%.4f'%(it, ls))
		im_gt = visualize.draw_label(batch[1][0])
		im_pred = visualize.draw_label(out[0].numpy())
		cv2.imshow('img', np.uint8(batch[0][0]))
		cv2.imshow('gt', im_gt)
		cv2.imshow('pred', im_pred)
		cv2.waitKey(5)

	if it%config.save_interval==0 and it>0:
		saver.save('./model/%d.ckpt'%it)

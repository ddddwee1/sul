import layers2 as L 
L.set_gpu('0')

import modeleag as M 
import tensorflow as tf 

class network(M.Model):
	def initialize(self):
		self.c1 = M.ConvLayer(7, 32, stride=2, activation=M.PARAM_RELU)
		self.c2 = M.ConvLayer(5, 64, stride=2, activation=M.PARAM_RELU)
		self.c3 = M.ConvLayer(5, 64, stride=1, activation=M.PARAM_RELU)
		self.c4 = M.ConvLayer(5, 128, stride=2, activation=M.PARAM_RELU)
		self.c5 = M.ConvLayer(5, 128, stride=1, activation=M.PARAM_RELU)
		self.c6 = M.ConvLayer(3, 128, stride=2, activation=M.PARAM_RELU)
		self.c7 = M.ConvLayer(3, 256, stride=1, activation=M.PARAM_RELU)
		self.c8 = M.ConvLayer(3, 256, stride=1, activation=M.PARAM_RELU)
		self.c9 = M.ConvLayer(1, 5, stride=1)

	def forward(self, x):
		x = self.c1(x)
		x = self.c2(x)
		x = self.c3(x)
		x = self.c4(x)
		x = self.c5(x)
		x = self.c6(x)
		x = self.c7(x)
		x = self.c8(x)
		x = self.c9(x)
		return x 

def confLoss(y, label):
	weight = tf.stop_gradient(tf.abs(tf.sigmoid(y) - label))

	loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=y) * weight
	loss = tf.reduce_sum(loss) / tf.reduce_sum(weight)

	return loss 

def regressLoss(y, label, conf):
	pred_xy = y[:,:,:,:2]
	pred_wh = y[:,:,:,2:]

	label_xy = label[:,:,:,:2]
	label_wh = label[:,:,:,2:]

	# label_wh = tf.log(label_wh)
	
	loss_xy = tf.square(label_xy - pred_xy) * conf
	loss_xy = tf.reduce_sum(loss_xy) / tf.reduce_sum(conf)

	loss_wh = tf.square(label_wh - pred_wh) * conf
	loss_wh = tf.reduce_sum(loss_wh) / tf.reduce_sum(conf)

	loss = loss_xy + loss_wh

	return loss_xy, loss_wh

def lossFunc(x, conf, geo, model):
	with tf.GradientTape() as tape:
		y = model(x)
		pred_conf = y[:,:,:,0:1]
		pred_geo = y[:,:,:,1:]

		conf_ls = confLoss(pred_conf, conf)
		ls_xy, ls_wh = regressLoss(pred_geo, geo, conf)

		losses = [conf_ls, ls_xy, ls_wh]
		weights = [10, 0.1, 1]
		loss_total = sum([weights[i]*losses[i] for i in range(len(losses))])
	return losses, loss_total, tape, pred_conf, pred_geo

def applyGrad(loss, model, optim, tape):
	grad = tape.gradient(loss, model.variables)
	optim.apply_gradients(zip(grad, model.variables))


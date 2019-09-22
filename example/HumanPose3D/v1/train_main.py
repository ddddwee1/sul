import model3 as M 
import network 
import tensorflow as tf 
import numpy as np 
import datareader
from tqdm import tqdm 

class modelBundle(M.Model):
	def initialize(self):
		self.refine2D = network.Refine2dNet()
		self.depth3D = network.DepthEstimator()
		# self.dis2D = network.Discriminator2D()
		# self.dis3D = network.Discriminator3D()

def grad_loss(x, model):
	x_2d, x_depth, mask = x 
	with tf.GradientTape() as tape:
		pred_2d = model.refine2D(x_2d * mask )
		pred_3d = model.depth3D(x_2d * mask )
		ls_2d = tf.reduce_mean(tf.square(pred_2d - x_2d))
		ls_3d = tf.reduce_mean(tf.square(pred_3d - x_depth))
		ls = ls_2d + ls_3d
	grads = tape.gradient(ls, model.trainable_variables)
	return grads, [ls_2d, ls_3d]

net = modelBundle()
# load last checkpoint
optim = tf.optimizers.Adam(0.0001)
saver = M.Saver(net)
saver.restore('./model/')

# create data reader
reader = datareader.DataReader(64, temp=32)

# start training
MAXITER = 50000

meter2d = M.EMAMeter(0.5)
meter3d = M.EMAMeter(0.5)
bar = tqdm(range(MAXITER+1))
for i in bar:
	batch = reader.get_next()
	grads, lss = grad_loss(batch, net)
	optim.apply_gradients(zip(grads, net.trainable_variables))

	ls2d = meter2d.update(lss[0].numpy())
	ls3d = meter3d.update(lss[1].numpy())

	bar.set_description('2D:%.4f 3D:%.4f'%(ls2d, ls3d))

	if i%2000==0 and i>0:
		saver.save('./model/%d.ckpt'%i)

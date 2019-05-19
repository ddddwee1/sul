import tensorflow as tf 
import model as M 
import numpy as np 
import layers as L 

def spconv(mod, dw_chn, outchn, stride, activation=M.PARAM_LRELU):
	mod.convLayer(1, dw_chn, activation=activation, batch_norm=True, usebias=False)
	mod.dwconvLayer(3, 1, stride=stride, activation=activation, batch_norm=True, usebias=False)
	mod.convLayer(1, outchn, batch_norm=True, usebias=False)

def ResUnit(mod, numblk, dw_chn, outchn):
	for i in range(numblk):
		identity = mod.get_current_layer()
		spconv(mod, dw_chn, outchn, 1, M.PARAM_LRELU)
		mod.sum(identity)

def build_net(mod):
	mod.convLayer(3, 64, stride=2, activation=M.PARAM_LRELU, batch_norm=True, usebias=False)
	ResUnit(mod, 2, 64, 64)
	spconv(mod, 128, 64, 2)
	ResUnit(mod, 8, 128, 64)
	spconv(mod, 256, 128, 2)
	ResUnit(mod, 16, 256, 128)
	spconv(mod, 512, 128, 2)
	ResUnit(mod, 4, 256, 128)
	mod.convLayer(1, 512, activation=M.PARAM_LRELU, batch_norm=True, usebias=False)
	mod.convLayer(7, 512, usebias=False, batch_norm=True, pad='VALID')
	mod.flatten()
	mod.fcLayer(256)

imgholder = tf.placeholder(tf.float32, [1, 112,112, 3])
mod = M.Model(imgholder)
build_net(mod)

run_meta = tf.RunMetadata()
with tf.Session() as sess:
	opts = tf.profiler.ProfileOptionBuilder.float_operation()
	flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)
	print('TF stats gives',flops.total_float_ops)

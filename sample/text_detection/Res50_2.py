import model as M 
import tensorflow as tf 
import numpy as np 

def build_model(is_training=False):
	with tf.variable_scope('Res50'):
		mod = M.Model(self.img_holder)
		mod.set_bn_training(is_training)
		# 64x64
		mod.convLayer(7,64,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
		mod.res_block(256,stride=2,activation=M.PARAM_LRELU)
		mod.res_block(256,stride=1,activation=M.PARAM_LRELU)
		c1 = mod.res_block(256,stride=1,activation=M.PARAM_LRELU)
		# 32x32
		mod.res_block(512,stride=2,activation=M.PARAM_LRELU)
		mod.res_block(512,stride=1,activation=M.PARAM_LRELU)
		mod.res_block(512,stride=1,activation=M.PARAM_LRELU)
		c2 = mod.res_block(512,stride=1,activation=M.PARAM_LRELU)
		# 16x16
		mod.res_block(1024,stride=2,activation=M.PARAM_LRELU)
		mod.res_block(1024,stride=1,activation=M.PARAM_LRELU)
		mod.res_block(1024,stride=1,activation=M.PARAM_LRELU)
		c3 = mod.res_block(1024,stride=1,activation=M.PARAM_LRELU)
		# 8x8
		mod.res_block(2048,stride=2,activation=M.PARAM_LRELU)
		mod.res_block(2048,stride=1,activation=M.PARAM_LRELU)
		c4 = mod.res_block(2048,stride=1,activation=M.PARAM_LRELU)
	return c4,c3,c2,c1

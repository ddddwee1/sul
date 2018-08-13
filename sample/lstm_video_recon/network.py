import tensorflow as tf 
import model as M 

def conv_layers(inp,reuse=False):
	with tf.variable_scope('enc',reuse=reuse):
		mod = M.Model(inp)
		mod.set_bn_training(False)
		mod.convLayer(7,16,stride=2,activation=M.PARAM_LRELU,batch_norm=True) #128
		mod.convLayer(5,32,stride=2,activation=M.PARAM_LRELU,batch_norm=True) # 64
		mod.convLayer(5,32,stride=2,activation=M.PARAM_LRELU,batch_norm=True) # 32
		mod.res_block(32,activation=M.PARAM_LRELU)
		mod.SelfAttention(8,residual=True)
		mod.convLayer(5,32,stride=2,activation=M.PARAM_LRELU,batch_norm=True) # 16
		mod.res_block(32,activation=M.PARAM_LRELU)
		mod.res_block(32,activation=M.PARAM_LRELU)
		mod.convLayer(5,32,stride=2,activation=M.PARAM_LRELU,batch_norm=True) # 8
		mod.res_block(32,activation=M.PARAM_LRELU)
		mod.res_block(32,activation=M.PARAM_LRELU)
		mod.convLayer(5,32,stride=2,activation=M.PARAM_LRELU,batch_norm=True) # 4
		mod.flatten()
	return mod.get_current_layer()

def deconv_layers(inp,reuse=False):
	with tf.variable_scope('dec',reuse=reuse):
		mod = M.Model(inp)
		mod.set_bn_training(False)
		mod.reshape([-1,4,4,32])
		mod.deconvLayer(5,32, stride=2, activation=M.PARAM_LRELU,batch_norm=True) # 8
		mod.res_block(32,activation=M.PARAM_LRELU)
		mod.res_block(32,activation=M.PARAM_LRELU)
		mod.SelfAttention(8)
		mod.deconvLayer(5,32, stride=2,activation=M.PARAM_LRELU,batch_norm=True) # 16
		mod.res_block(32,activation=M.PARAM_LRELU)
		feat = mod.deconvLayer(5,32, stride=2,activation=M.PARAM_LRELU,batch_norm=True) # 32
		mod.res_block(32,activation=M.PARAM_LRELU)
		A = mod.convLayer(5,3,activation=M.PARAM_SIGMOID)
		mod.set_current(feat)
		mod.res_block(32,activation=M.PARAM_LRELU)
		mod.res_block(32,activation=M.PARAM_LRELU)
		C = mod.convLayer(5,3,activation=M.PARAM_TANH)
		A = tf.image.resize_images(A,(256,256))
		C = tf.image.resize_images(C,(256,256))
	return A,C

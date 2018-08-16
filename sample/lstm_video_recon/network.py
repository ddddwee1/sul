import tensorflow as tf 
import model as M 

bn_training = True

def conv_layers(inp,reuse=False):
	global bn_training
	with tf.variable_scope('enc',reuse=reuse):
		mod = M.Model(inp)
		mod.set_bn_training(bn_training)
		mod.convLayer(7,16,stride=2,activation=M.PARAM_LRELU,batch_norm=True) #128
		mod.convLayer(5,32,stride=2,activation=M.PARAM_LRELU,batch_norm=True) # 64
		mod.convLayer(5,64,stride=2,activation=M.PARAM_LRELU,batch_norm=True) # 32
		mod.SelfAttention(8,residual=True)
		mod.convLayer(5,64,stride=2,activation=M.PARAM_LRELU,batch_norm=True) # 16
		mod.res_block(64,activation=M.PARAM_LRELU)
		mod.res_block(64,activation=M.PARAM_LRELU)
		mod.convLayer(5,128,stride=2,activation=M.PARAM_LRELU,batch_norm=True) # 8
		mod.res_block(128,activation=M.PARAM_LRELU)
		mod.res_block(128,activation=M.PARAM_LRELU)
		mod.SelfAttention(32)
		mod.convLayer(5,128,stride=2,activation=M.PARAM_LRELU,batch_norm=True) # 4
		mod.res_block(128,activation=M.PARAM_LRELU)
		mod.res_block(128,activation=M.PARAM_LRELU)
		mod.flatten()
	return mod.get_current_layer()

def deconv_layers(inp,reuse=False):
	global bn_training
	with tf.variable_scope('dec',reuse=reuse):
		mod = M.Model(inp)
		mod.set_bn_training(bn_training)
		mod.reshape([-1,4,4,128])
		mod.deconvLayer(5,128, stride=2, activation=M.PARAM_LRELU,batch_norm=True) # 8
		mod.res_block(128,activation=M.PARAM_LRELU)
		mod.res_block(128,activation=M.PARAM_LRELU)
		mod.SelfAttention(32)
		mod.deconvLayer(5,64, stride=2,activation=M.PARAM_LRELU,batch_norm=True) # 16
		mod.res_block(64,activation=M.PARAM_LRELU)
		mod.res_block(64,activation=M.PARAM_LRELU)
		mod.deconvLayer(5,64,stride=2,activation=M.PARAM_LRELU,batch_norm=True) #32
		mod.res_block(64,activation=M.PARAM_LRELU)
		feat = mod.deconvLayer(5,32, stride=2,activation=M.PARAM_LRELU,batch_norm=True) # 64
		mod.res_block(32,activation=M.PARAM_LRELU)
		mod.deconvLayer(5, 32, stride=2, activation=M.PARAM_LRELU,batch_norm=True) #128
		A = mod.convLayer(5,3,activation=M.PARAM_SIGMOID)
		mod.set_current(feat)
		mod.res_block(32,activation=M.PARAM_LRELU)
		mod.deconvLayer(5,32,stride=2,activation=M.PARAM_LRELU,batch_norm=True) # 128
		mod.res_block(32,activation=M.PARAM_LRELU)
		mod.deconvLayer(5,16,stride=2,activation=M.PARAM_LRELU,batch_norm=True) #256
		C = mod.convLayer(5,3,activation=M.PARAM_TANH)
		A = tf.image.resize_images(A,(256,256))
		# C = tf.image.resize_images(C,(256,256))
	return A,C

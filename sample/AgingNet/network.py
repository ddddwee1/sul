import tensorflow as tf 
from dyn import model_attention as Model
import model as M

bn_training = True

reuse_att = False
reuse_enc = False
reuse_gen = False
reuse_dis = False
reuse_feat = False
reuse_dis2 = False
reuse_dis_f = False
reuse_genatt = False
reuse_agecls = False
reuse_agecls_r = False
reuse_age_enc = {}

blknum = 0

def feat_encoder(inp):
	global reuse_enc,bn_training
	with tf.variable_scope('encoder',reuse=reuse_enc):
		mod = Model(inp)
		mod.set_bn_training(bn_training)
		mod.convLayer(7,16,stride=2,activation=M.PARAM_LRELU,batch_norm=True) #128
		mod.maxpoolLayer(3,stride=2) # 64
		mod.convLayer(5,64,stride=2,activation=M.PARAM_LRELU,batch_norm=True) # 32
		block(mod,256,2) # 16
		block(mod,256,1)
		block(mod,256,1)
		mod.SelfAttention(32)
		block(mod,256,2) # 8
		block(mod,256,1) 
		block(mod,256,1)
		mod.SelfAttention(32)
		block(mod,512,2) #4
		block(mod,512,1)
		block(mod,512,1)
		reuse_enc = True
	return mod.get_current_layer()

def block(mod,output,stride):
	global blknum
	with tf.variable_scope('block'+str(blknum)):
		inp = mod.get_current().get_shape().as_list()[-1]
		aa = mod.get_current()
		if inp==output:
			if stride==1:
				l0 = mod.get_current()
			else:
				l0 = mod.maxpoolLayer(stride)
		else:
			l0 = mod.convLayer(1,output,activation=M.PARAM_RELU,stride=stride)
		mod.set_current_layer(aa)
		mod.batch_norm()
		mod.activate(M.PARAM_RELU)
		mod.convLayer(1,output//4,activation=M.PARAM_RELU,batch_norm=True,stride=stride)
		mod.convLayer(3,output//4,activation=M.PARAM_RELU,batch_norm=True)
		mod.convLayer(1,output)
		mod.sum(l0)
		blknum+=1
	return mod

def age_encoder(inp,ind):
	global reuse_age_enc
	name = 'decoder'+str(ind)
	if not name in reuse_age_enc:
		reuse = False
	else:
		reuse = True
	with tf.variable_scope(name,reuse=reuse):
		mod = Model(inp)
		mod.fcLayer(2*2*512,activation=M.PARAM_RELU)
		mod.SelfAttention(is_fc=True,residual=True)
		reuse_age_enc[name] = 1
	return mod.get_current_layer()

def generator(inp):
	global reuse_gen,bn_training
	with tf.variable_scope('generator',reuse=reuse_gen):
		mod = Model(inp)
		mod.set_bn_training(bn_training)
		mod.reshape([-1,2,2,512])
		mod.deconvLayer(3,256,stride=2,activation=M.PARAM_LRELU,batch_norm=True) #8
		mod.deconvLayer(3,128,stride=2,activation=M.PARAM_LRELU,batch_norm=True) #16
		mod.SelfAttention(32)
		mod.deconvLayer(5,64,stride=2,activation=M.PARAM_LRELU,batch_norm=True) #32
		mod.deconvLayer(5,32,stride=2,activation=M.PARAM_LRELU,batch_norm=True) #64
		mod.deconvLayer(5,16,stride=2,activation=M.PARAM_LRELU,batch_norm=True) #128
		mod.deconvLayer(5,3,activation=M.PARAM_TANH) #output
		reuse_gen = True
	return mod.get_current_layer()

def discriminator(inp,age_size):
	global reuse_dis,bn_training,blknum
	blknum = 0
	with tf.variable_scope('discriminator',reuse=reuse_dis):
		mod = Model(inp)
		mod.set_bn_training(bn_training)
		mod.convLayer(7,16,stride=2,activation=M.PARAM_LRELU,batch_norm=True) # 64
		mod.convLayer(5,32,stride=2,activation=M.PARAM_LRELU,batch_norm=True) # 32
		mod.SelfAttention(4)
		feat = mod.convLayer(5,64,stride=2,activation=M.PARAM_LRELU) # 16
		mod.batch_norm()
		mod.convLayer(3,128,stride=2,activation=M.PARAM_LRELU,batch_norm=True) # 8
		adv = mod.convLayer(3,1)
		mod.set_current_layer(feat)
		block(mod,128,1)
		block(mod,128,2) # 8
		block(mod,256,1)
		mod.SelfAttention(32)
		block(mod,256,1)
		block(mod,256,2) # 4
		block(mod,256,1)
		mod.SelfAttention(32)
		block(mod,256,2) # 2
		block(mod,256,1)
		mod.flatten()
		mod.fcLayer(512,activation=M.PARAM_LRELU)
		age = mod.fcLayer(age_size)
		reuse_dis = True
	return adv,age

def discriminator_feature(inp):
	global reuse_dis2,bn_training
	with tf.variable_scope('discriminator_feature',reuse=reuse_dis2):
		mod = Model(inp)
		mod.set_bn_training(bn_training)
		mod.fcLayer(256,activation=M.PARAM_LRELU,batch_norm=True)
		mod.fcLayer(64,activation=M.PARAM_LRELU,batch_norm=True)
		mod.fcLayer(1)
		reuse_dis2 = True
	return mod.get_current_layer()

def attention_blk(features):
	global reuse_att
	with tf.variable_scope('attention_blk',reuse=reuse_att):
		# get q0
		f_dim = features.get_shape().as_list()[-1]
		q0 = tf.get_variable('q0',[1,f_dim],initializer=tf.random_normal_initializer(),dtype=tf.float32)
		BSIZE = tf.shape(features[0])
		q0 = tf.tile(q0,[BSIZE,1])

		mod = Model(q0)
		mod.QAttention(features)
		mod.fcLayer(f_dim,activation=M.PARAM_TANH)
		mod.QAttention(features)
		reuse_att = True
	return mod.get_current_layer()

def discriminator_f(inp,id_num):
	global reuse_dis_f,bn_training
	with tf.variable_scope('dis_f',reuse=reuse_dis_f):
		mod = Model(inp)
		mod.set_bn_training(bn_training)
		mod.flatten()
		feat = mod.get_current_layer()
		mod.fcLayer(512,activation=M.PARAM_LRELU,batch_norm=True)
		mod.fcLayer(256,activation=M.PARAM_LRELU,batch_norm=True)
		adv = mod.fcLayer(1)
		mod.set_current_layer(feat)
		ip = mod.fcLayer(id_num)
		reuse_dis_f = True
	return adv,ip

def generator_att(inp):
	global reuse_genatt,bn_training
	with tf.variable_scope('gen_att',reuse=reuse_genatt):
		mod = Model(inp)
		mod.set_bn_training(bn_training)
		mod.deconvLayer(3,512,stride=2,activation=M.PARAM_LRELU,batch_norm=True) #4
		mod.deconvLayer(3,256,stride=2,activation=M.PARAM_LRELU,batch_norm=True) #8
		mod.deconvLayer(3,128,stride=2,activation=M.PARAM_LRELU,batch_norm=True) #16
		mod.SelfAttention(32)
		mod.deconvLayer(5,64,stride=2,activation=M.PARAM_LRELU,batch_norm=True) #32
		mod.deconvLayer(5,32,stride=2,activation=M.PARAM_LRELU,batch_norm=True) #64
		feat = mod.deconvLayer(5,16,stride=2,activation=M.PARAM_LRELU,batch_norm=True) #128
		A = mod.convLayer(5,3,activation=M.PARAM_SIGMOID) #output_attention
		mod.set_current_layer(feat)
		C = mod.convLayer(5,3,activation=M.PARAM_TANH)
		reuse_genatt = True
	return A,C

def age_classify(inp,age_size):
	global reuse_agecls, bn_training
	with tf.variable_scope('age_cls',reuse=reuse_agecls):
		mod = Model(inp)
		mod.set_bn_training(bn_training)
		mod.flatten()
		mod.fcLayer(512,activation=M.PARAM_LRELU)
		mod.fcLayer(256,activation=M.PARAM_LRELU)
		mod.fcLayer(age_size)
		reuse_agecls = True
	return mod.get_current_layer()

def age_classify_r(inp,age_size):
	global reuse_agecls, bn_training
	with tf.variable_scope('age_cls',reuse=reuse_agecls):
		mod = Model(inp)
		mod.set_bn_training(bn_training)
		mod.gradient_flip_layer()
		mod.flatten()
		mod.fcLayer(512,activation=M.PARAM_LRELU)
		mod.fcLayer(256,activation=M.PARAM_LRELU)
		mod.fcLayer(age_size)
		reuse_agecls = True
	return mod.get_current_layer()
import model as M 
import numpy as np 
import tensorflow as tf 
# import scipy.io as sio
import pickle
import layers as L
import utils

block_num = 0

def res_block(mod,kernel_size,channels,stride,with_short):
	global block_num
	chn1, chn2, chn3 = channels
	with tf.variable_scope('block_'+str(block_num)):
		inputLayer = mod.get_current()
		mod.convLayer(1,chn1,stride=stride,batch_norm=True,activation=M.PARAM_RELU)
		mod.convLayer(kernel_size,chn2,batch_norm=True,activation=M.PARAM_RELU)
		branch = mod.convLayer(1,chn3,batch_norm=True)
		# Shortcut
		mod.set_current(inputLayer)
		if with_short:
			mod.convLayer(1,chn3,stride=stride,batch_norm=True)
		mod.sum(branch)
		mod.activation(M.PARAM_RELU)
	block_num += 1

def resnet(inputLayer,stage5=False):
	with tf.variable_scope('resnet'):
		mod = M.Model(inputLayer)
		mod.set_bn_training(False)
		# 1
		mod.pad(3)
		mod.convLayer(7,64,stride=2,pad='VALID',batch_norm=True,activation=M.PARAM_RELU)
		mod.maxpoolLayer(3,stride=2)
		C1 = mod.get_current_layer()
		# 2
		res_block(mod,3,[64,64,256],1,True)
		res_block(mod,3,[64,64,256],1,False)
		res_block(mod,3,[64,64,256],1,False)
		C2 = mod.get_current_layer()
		# 3
		res_block(mod,3,[128,128,512],2,True)
		res_block(mod,3,[128,128,512],1,False)
		res_block(mod,3,[128,128,512],1,False)
		res_block(mod,3,[128,128,512],1,False)
		C3 = mod.get_current_layer()
		# 4
		res_block(mod,3,[256,256,1024],2,True)
		for i in range(22):
			res_block(mod,3,[256,256,1024],1,False)
		C4 = mod.get_current_layer()
		# 5
		if stage5:
			res_block(mod,3,[512,512,2048],2,True)
			res_block(mod,3,[512,512,2048],1,False)
			res_block(mod,3,[512,512,2048],1,False)
			C5 = mod.get_current_layer()
		else:
			C5 = None
	return mod,C1,C2,C3,C4,C5

def get_rpn_layers(c2,c3,c4,c5):
	P5 = L.conv2D(c5,1,256,'P5')
	P4 = L.upSampling(P5,2,'U5') + L.conv2D(c4,1,256,'P4')
	P3 = L.upSampling(P4,2,'U4') + L.conv2D(c3,1,256,'P3')
	P2 = L.upSampling(P3,2,'U3') + L.conv2D(c2,1,256,'P2')
	P2 = L.conv2D(P2,3,256,'P22')
	P3 = L.conv2D(P3,3,256,'P32')
	P4 = L.conv2D(P4,3,256,'P42')
	P5 = L.conv2D(P5,3,256,'P52')
	P6 = L.maxpooling(P5,1,2,'P6')
	return P2,P3,P4,P5,P6

def get_assign_tensors():
	v = M.get_all_vars()
	print(len(v))

	with open('buffer_weights.pickle','rb') as f:
		fmat = pickle.load(f)

	assign_tensor = []
	# fmat = sio.loadmat('layer1')
	for i in range(640):
		buff = fmat[str(i)]
		# buff = np.float32(buff)
		if len(buff.shape)==2:
			buff = buff[0]
		bufftensor = tf.assign(v[i],buff)
		assign_tensor.append(bufftensor)
	return assign_tensor

class RPN():
	def __init__(self,anchor_stride,anchors_per_loc, depth):
		self.anchor_stride = anchor_stride
		self.anchors_per_loc = anchors_per_loc
		self.reuse = False

	def deploy(self,input_layer):
		with tf.variable_scope('RPN_',reuse=self.reuse):
			shared_feature = L.conv2D(input_layer,3,512,stride=self.anchor_stride,name='share_conv')
			shared_feature = L.relu(shared_feature,'share_relu')
			rpn_bf_logits = L.conv2D(shared_feature,1,2*self.anchors_per_loc,'bf')
			rpn_bf_logits = tf.reshape(rpn_bf_logits,[tf.shape(rpn_bf_logits)[0],-1,2])
			rpn_bf_prob = tf.nn.softmax(rpn_bf_logits)
			rpn_bbox = L.conv2D(shared_feature,1,4*self.anchors_per_loc,'bbox')
			rpn_bbox = tf.reshape(rpn_bbox,[tf.shape(rpn_bbox[0],-1,2)])
		self.reuse = True
		return rpn_bf_logits, rpn_bf_prob, rpn_bbox

def get_rpn_results(RPN_features,config):
	rpn = RPN(config.RPN_ANCHOR_STRIDE,len(config.RPN_ANCHOR_RATIOS),256)
	results_logits = []
	results_prob = []
	results_bbox = []
	for p in RPN_features:
		buff_bf_logit , buff_bf_prob , buff_bbox = rpn.deploy(p)
		results_logits.append(buff_bf_logit)
		results_prob.append(buff_bf_prob)
		results_bbox.append(buff_bbox)
	results_logits = tf.concat(results_logits,1)
	results_prob = tf.concat(results_prob, 1)
	results_bbox = tf.concat(results_bbox, 1)
	return results_logits,results_prob,results_bbox

inpholder = tf.placeholder(tf.float32,[None,1024,1024,3])
_,c1,c2,c3,c4,c5 = resnet(inpholder,True)
P2,P3,P4,P5,P6 = get_rpn_layers(c2,c3,c4,c5)
RPN_features = [P2,P3,P4,P5,P6]
config = utils.get_config()
anchors = utils.generate_all_anchors(config.RPN_ANCHOR_SCALES,config.RPN_ANCHOR_RATIOS,config.BACKBONE_SHAPES,config.BACKBONE_STRIDES,config.RPN_ANCHOR_STRIDE)
rpn_logits, rpn_prob, rpn_bbox = get_rpn_results(RPN_features,config)


assign_tensor = get_assign_tensors()
with tf.Session() as sess:
	writer = tf.summary.FileWriter('./logs/',sess.graph)
	M.loadSess('./model/',sess,init=True)
	sess.run(assign_tensor)
	img = np.ones([1,1024,1024,3])
	res = sess.run(P6,feed_dict={inpholder:img})
	print(res)
	print(res.shape)


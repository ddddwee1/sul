import tensorflow as tf 
import model as M 
import numpy as np 

def ResBlock(mod, outchn, stride, sc=False):
	x = mod.get_current_layer()
	mod.batch_norm()
	mod.convLayer(3, outchn, activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
	mod.convLayer(3, outchn, stride=stride, usebias=False, batch_norm=True)
	branch = mod.get_current_layer()

	mod.set_current_layer(x)
	if sc:
		mod.convLayer(1, outchn, stride=stride, usebias=False, batch_norm=True)
	mod.sum(branch)

def HeadBlock(mod, outchn):
	mod.convLayer(3, outchn, usebias=False, activation=M.PARAM_LRELU, batch_norm=True)

def ResNet(inp, channel_list, blocknum_list, emb_size):
	mod = M.Model(inp)
	HeadBlock(mod, channel_list[0])
	for num,chn in zip(blocknum_list, channel_list[1:]):
		for i in range(num):
			ResBlock(mod, chn, 2 if i==0 else 1, i==0)
	mod.flatten()
	mod.batch_norm()
	mod.dropout(0.6)
	mod.fcLayer(emb_size, batch_norm=True)
	return mod.get_current_layer()

def ArcLayer(layer, label, cls_num, emb_size, m1, m2, m3):
	layer = tf.nn.l2_normalize(layer, axis=1)
	W = tf.get_variable('classify_weight', [emb_size, cls_num],initializer=tf.contrib.layers.xavier_initializer(),dtype=dtype)
	W = tf.nn.l2_normalize(W, axis=0)
	x = tf.matmul(layer, W)

	if not (m1==1.0 and m2==0.0):
		t = tf.gather_nd(x, indices=tf.where(label>0))
		t = tf.math.acos(t)
		if m1!=1.0:
			t = t*m1
		if m2!=0.0:
			t = t+m2
		t = tf.math.cos(t)
		t = tf.expand_dims(t, axis=1)
		x = x*(1 - label) + t*label

	if m3!=0.0:
		x = x - label*m3 
	return x 

def split_data(data, num_devices):
	res = []
	length = len(data[0])
	len_split = length//num_devices
	for i in range(num_devices):
		buff = []
		for j in range(len(data)):
			buff.append(data[j][i*len_split: min(length, i*len_split+len_split)])
		res.append(buff)
	return res 

BSIZE = 4*128
data_reader = datareader.DataReader('img_list_v1.txt', BSIZE)
cls_num = data_reader.max_label + 1
channel_list = [64,64,128,256,512]
blocknum_list = [3, 4, 14, 3]
emb_size = 512
LR = 0.1
GPUS = ['/.gpu:0','/.gpu:1','/.gpu:2','/.gpu:3']


with tf.device('/.cpu:0'):
	# build graph 
	imgholder = tf.placeholder(tf.float32, [len(GPUS), None, 112,112, 3])
	labelholder = tf.placeholder(tf.float32, [len(GPUS), None, cls_num])
	lrholder = tf.placeholder(tf.float32, [])

	optim = tf.train.MomentumOptimizer(lrholder, 0.9)
	all_grads = []
	accs = []
	for idx,g in eumerate(GPUS):
		with tf.device(g):
			emb = ResNet(imgholder[idx], channel_list, blocknum_list, emb_size)
			logits = ArcLayer(emb, labelholder[idx], cls_num, emb_size, 1.0, 0.5, 0.0)

			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labelholder[idx]))
			acc = M.accuracy(logits, tf.argmax(labelholder[idx], -1))
			accs.append(acc)

			tf.get_variable_scope().reuse_variables()

			grads = optim.compute_gradients(loss, M.get_trainable_vars())

			all_grads.append(grads)

	grads_ = []
	for i in zip(*all_grads):
		grads_.append(sum(i) / len(i))
	train_op = optim.apply_gradients(grads_, M.get_trainable_vars())
	acc = tf.reduce_mean(accs)

saver = tf.train.Saver()

with tf.Session() as sess:
	M.loadSess('./model/', sess=sess, init=True)
	for ep in range(EPOCH):
		if ep in [6,10,14]:
			LR *= 0.1
		for it in range(data_reader.iter_per_epoch):
			batch = data_reader.get_next()
			batch_distribute = split_data(batch, len(GPUS))

			ls, ac, _ = sess.run([loss, acc, train_op], feed_dict={imgholder:batch_distribute[0], 
																	labelholder:batch_distribute[1],
																	lrholder:LR})

			if i%10==0:
				print('Epoch:%d\tIter:%d\tLoss:%.4f\tAcc:%.4f\tLR:%f'%(ep, it, ls, ac, LR))

			if it%5000==0 and it>0:
				saver.save('./model/%d_%d.ckpt'%(ep,it))
		saver.save('./model/%d_%d.ckpt'%(ep,it))

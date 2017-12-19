import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import model as M
import numpy as np 
import tensorflow as tf 
import cv2 
import random
from collections import deque

BSIZE = 10
ZDIM = 64
CLASS = 100

def localpath_le(inp):
	with tf.variable_scope('local_le'):
		mod = M.Model(inp,[None,40,40,3])
		conv0 = mod.convLayer(3,64,activation=M.PARAM_LRELU,batch_norm=True)
		conv1 = mod.convLayer(3,128,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
		conv2 = mod.convLayer(3,256,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
		mod.convLayer(3,512,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
		mod.deconvLayer(3,256,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
		mod.concat_to_current(conv2)
		mod.deconvLayer(3,128,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
		mod.concat_to_current(conv1)
		mod.deconvLayer(3,64,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
		mod.concat_to_current(conv0)
		mod.convLayer(3,64,activation=M.PARAM_LRELU,batch_norm=True)
		mod.convLayer(3,3,activation=M.PARAM_LRELU)
	return mod.get_current_layer()

def localpath_re(inp):
	with tf.variable_scope('local_re'):
		mod = M.Model(inp,[None,40,40,3])
		conv0 = mod.convLayer(3,64,activation=M.PARAM_LRELU,batch_norm=True)
		conv1 = mod.convLayer(3,128,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
		conv2 = mod.convLayer(3,256,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
		mod.convLayer(3,512,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
		mod.deconvLayer(3,256,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
		mod.concat_to_current(conv2)
		mod.deconvLayer(3,128,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
		mod.concat_to_current(conv1)
		mod.deconvLayer(3,64,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
		mod.concat_to_current(conv0)
		mod.convLayer(3,64,activation=M.PARAM_LRELU,batch_norm=True)
		mod.convLayer(3,3,activation=M.PARAM_LRELU)
	return mod.get_current_layer()

def localpath_mth(inp):
	with tf.variable_scope('local_mth'):
		mod = M.Model(inp,[None,32,48,3])
		conv0 = mod.convLayer(3,64,activation=M.PARAM_LRELU,batch_norm=True)
		conv1 = mod.convLayer(3,128,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
		conv2 = mod.convLayer(3,256,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
		mod.convLayer(3,512,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
		mod.deconvLayer(3,256,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
		mod.concat_to_current(conv2)
		mod.deconvLayer(3,128,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
		mod.concat_to_current(conv1)
		mod.deconvLayer(3,64,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
		mod.concat_to_current(conv0)
		mod.convLayer(3,64,activation=M.PARAM_LRELU,batch_norm=True)
		mod.convLayer(3,3,activation=M.PARAM_LRELU)
	return mod.get_current_layer()

def localpath_nse(inp):
	with tf.variable_scope('local_nse'):
		mod = M.Model(inp,[None,32,40,3])
		conv0 = mod.convLayer(3,64,activation=M.PARAM_LRELU,batch_norm=True)
		conv1 = mod.convLayer(3,128,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
		conv2 = mod.convLayer(3,256,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
		mod.convLayer(3,512,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
		mod.deconvLayer(3,256,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
		mod.concat_to_current(conv2)
		mod.deconvLayer(3,128,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
		mod.concat_to_current(conv1)
		mod.deconvLayer(3,64,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
		mod.concat_to_current(conv0)
		mod.convLayer(3,64,activation=M.PARAM_LRELU,batch_norm=True)
		mod.convLayer(3,3,activation=M.PARAM_LRELU)
	return mod.get_current_layer()

def globalpath(inp,z,local):
	with tf.variable_scope('global_path'):
		mod = M.Model(inp,[None,128,128,3])
		conv0 = mod.convLayer(7,64,activation=M.PARAM_LRELU,batch_norm=True)
		conv1 = mod.convLayer(5,64,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
		conv2 = mod.convLayer(3,128,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
		conv3 = mod.convLayer(3,256,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
		conv4 = mod.convLayer(3,512,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
		mod.flatten()
		fc1 = mod.fcLayer(512,activation=M.PARAM_LRELU,batch_norm=True)
		fc2 = mod.fcLayer(256,activation=M.PARAM_LRELU,batch_norm=True)
		# global_class = mod.fcLayer(CLASS)
		mod.gradient_flip_layer()
		domain = mod.fcLayer(256,activation=M.PARAM_LRELU,batch_norm=True)
		domain = mod.fcLayer(256,activation=M.PARAM_LRELU,batch_norm=True)
		domain = mod.fcLayer(2)
		mod.set_current(fc2)
		mod.concat_feature([z,[None,ZDIM]])
		feat8 = mod.fcLayer(8*8*64,activation=M.PARAM_LRELU,batch_norm=True)
		feat8 = mod.construct([8,8,64])
		feat32 = mod.deconvLayer(3,32,stride=4,activation=M.PARAM_LRELU,batch_norm=True)
		feat64 = mod.deconvLayer(3,16,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
		feat128 = mod.deconvLayer(3,8,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
		mod.set_current(feat8)
		mod.concat_to_current(conv4)
		deconv0 = mod.deconvLayer(3,512,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
		mod.concat_to_current(conv3)
		mod.deconvLayer(3,256,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
		mod.concat_to_current(feat32)
		mod.concat_to_current(conv2)
		mod.concat_to_current([tf.image.resize_images(inp,(32,32)),[None,32,32,3]])
		mod.deconvLayer(3,128,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
		mod.concat_to_current(feat64)
		mod.concat_to_current(conv1)
		mod.concat_to_current([tf.image.resize_images(inp,(64,64)),[None,64,64,3]])
		mod.deconvLayer(3,64,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
		mod.concat_to_current(feat128)
		mod.concat_to_current(conv0)
		mod.concat_to_current([tf.image.resize_images(inp,(128,128)),[None,128,128,3]])
		mod.concat_to_current([local,[None,128,128,3]])
		mod.convLayer(5,64,activation=M.PARAM_LRELU,batch_norm=True)
		mod.convLayer(3,32,activation=M.PARAM_LRELU,batch_norm=True)
		mod.convLayer(3,3,activation=M.PARAM_LRELU)
	return mod.get_current_layer(),domain[0]

def fusion_locals(le,re,nse,mth):
	with tf.variable_scope('fusion_node'):
		padded_le = tf.pad(le,[[0,0],[24,64],[24,64],[0,0]])
		padded_re = tf.pad(re,[[0,0],[24,64],[64,24],[0,0]])
		padded_nse = tf.pad(nse,[[0,0],[48,48],[44,44],[0,0]])
		padded_mth = tf.pad(mth,[[0,0],[70,26],[40,40],[0,0]])
		ttl = tf.stack([padded_mth , padded_nse , padded_le , padded_re])
		return tf.reduce_max(ttl,axis=[0])

def discriminator(inp,reuse=False):
	with tf.variable_scope('dis',reuse=reuse):
		mod = M.Model(inp,[None,128,128,3])
		conv0 = mod.convLayer(7,64,activation=M.PARAM_LRELU,batch_norm=True)
		conv1 = mod.convLayer(5,64,stride=2,activation=M.PARAM_LRELU,batch_norm=True)#64
		conv2 = mod.convLayer(3,128,stride=2,activation=M.PARAM_LRELU,batch_norm=True)#32
		conv3 = mod.convLayer(3,256,stride=2,activation=M.PARAM_LRELU,batch_norm=True)#16
		conv4 = mod.convLayer(3,512,stride=2,activation=M.PARAM_LRELU,batch_norm=True)#8
		conv4 = mod.convLayer(3,512,activation=M.PARAM_LRELU,batch_norm=True)
		k = mod.convLayer(3,512,stride=2,activation=M.PARAM_LRELU,batch_norm=True)
		k = mod.convLayer(3,512,activation=M.PARAM_LRELU,batch_norm=True)
		branch = mod.flatten()
		branch = mod.fcLayer(CLASS)
		branch = mod.get_current_layer()
		mod.set_current(k)
		k = mod.transpose([1,2,0,3])
		k = mod.reshape([4,4,BSIZE*512,1])
		k = mod.get_current_layer()
		mod.set_current(conv4)
		conv4_trans = mod.transpose([1,2,0,3])
		conv4_trans = mod.reshape([1,8,8,BSIZE*512])
		mod.dwconvLayer(4,1,weight=k)
		mod.reshape([8,8,BSIZE,512])
		mod.transpose([2,0,1,3])
		mod.convLayer(1,512)
		mod.convLayer(3,512,activation=M.PARAM_LRELU,batch_norm=True)
		mod.flatten()
		mod.fcLayer(1)
		return mod.get_current_layer(),branch

blknum = 0

def block(mod,output):
	global blknum
	with tf.variable_scope('block'+str(blknum)):
		aa = mod.get_current()
		mod.convLayer(3,output*2,activation=M.PARAM_MFM,layerin=aa)
		mod.convLayer(3,output*2,activation=M.PARAM_MFM)
		mod.sum(aa)
		blknum+=1
	return mod

def res_18(inp):
	global blknum
	blknum = 0
	mod = M.Model(inp,[None,128,128,3])
	mod.convLayer(5,96,activation=M.PARAM_MFM)
	mod.maxpoolLayer(2)#64
	block(mod,48)
	mod.convLayer(1,96,activation=M.PARAM_MFM)
	mod.convLayer(3,192,activation=M.PARAM_MFM)
	mod.maxpoolLayer(2)
	block(mod,96)
	block(mod,96)
	mod.convLayer(1,192,activation=M.PARAM_MFM)
	mod.convLayer(3,384,activation=M.PARAM_MFM)
	mod.maxpoolLayer(2)
	block(mod,192)
	block(mod,192)
	block(mod,192)
	mod.convLayer(1,192,activation=M.PARAM_MFM)
	mod.convLayer(3,256,activation=M.PARAM_MFM)
	block(mod,128)
	block(mod,128)
	block(mod,128)
	block(mod,128)
	mod.convLayer(1,256,activation=M.PARAM_MFM)
	mod.convLayer(3,256,activation=M.PARAM_MFM)
	mod.maxpoolLayer(2)
	mod.flatten()
	mod.fcLayer(512)
	featurelayer = mod.get_current_layer()
	return featurelayer

def lcnn(inp,reuse=False):
	with tf.variable_scope('MainModel',reuse=reuse):
		featurelayer = res_18(inp)
	return featurelayer


def build_total_graph():
	global totalLoss,dis_loss,train_d,train_g,train_e,updd,updg,upde,profHolder,gtHolder,leHolder,reHolder,mthHolder,nseHolder,domainHolder,zHolder,clsHolder,x_fake,losscollection
	with tf.name_scope('ProfileImg'):
		profHolder = tf.placeholder(tf.float32,[None,128,128,3])
	with tf.name_scope('GTIMG'):
		gtHolder = tf.placeholder(tf.float32,[None,128,128,3])
	with tf.name_scope('LEIMG'):
		leHolder = tf.placeholder(tf.float32,[None,40,40,3])
	with tf.name_scope('REIMG'):
		reHolder = tf.placeholder(tf.float32,[None,40,40,3])
	with tf.name_scope('MTHIMG'):
		mthHolder = tf.placeholder(tf.float32,[None,32,48,3])
	with tf.name_scope('NSEIMG'):
		nseHolder = tf.placeholder(tf.float32,[None,32,40,3])
	with tf.name_scope('Z'):
		zHolder = tf.placeholder(tf.float32,[None,ZDIM])
	with tf.name_scope('domain'):
		domainHolder = tf.placeholder(tf.int32,[None])
	with tf.name_scope('CLASS'):
		clsHolder = tf.placeholder(tf.int32,[None])

	nse = localpath_nse(nseHolder)
	mth = localpath_mth(mthHolder)
	le = localpath_le(leHolder)
	re = localpath_re(reHolder)
	fusion = fusion_locals(le,re,nse,mth)
	x_fake,domainlayer = globalpath(profHolder,zHolder,fusion)
	d_fake,c_fake = discriminator(x_fake)
	d_real,c_real = discriminator(gtHolder,reuse=True)
	f_fake = lcnn(x_fake)
	f_real = lcnn(gtHolder,reuse=True)

	with tf.name_scope('pixel_loss'):
		pix_loss = tf.reduce_mean(tf.abs(gtHolder-x_fake))
	with tf.name_scope('sym_loss'):
		x_left, x_right = tf.split(x_fake,2,axis=2)
		sym_loss = tf.reduce_mean(tf.abs(x_left-x_right))
	with tf.name_scope('dis_loss'):
		dis_true = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real,labels=tf.ones([BSIZE,1])))
		dis_false= tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake,labels=tf.zeros([BSIZE,1])))
		dis_loss = dis_true+dis_false
	with tf.name_scope('gen_loss'):
		gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake,labels=tf.ones([BSIZE,1])))
	with tf.name_scope('ip_loss'):
		ip_loss = tf.reduce_mean(tf.abs(f_real-f_fake))
	with tf.name_scope('tv_loss'):
		tv_loss = tf.reduce_mean(tf.image.total_variation(x_fake))/(128.0*128.0)
	with tf.name_scope('domain_loss'):
		domain_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=domainlayer,labels=domainHolder))
	with tf.name_scope('cls_loss'):
		cls_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=c_real,labels=clsHolder))
		# cls_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=c_fake,labels=clsHolder))


	vard = M.get_trainable_vars('dis')
	varg = M.get_trainable_vars('local_le') \
	+ M.get_trainable_vars('local_re') \
	+ M.get_trainable_vars('local_mth') \
	+ M.get_trainable_vars('local_nse')\
	+ M.get_trainable_vars('global_path')\
	+ M.get_trainable_vars('fusion_node')
	updd = M.get_update_ops('dis')
	updg = M.get_update_ops('local_le') \
	+ M.get_update_ops('local_re') \
	+ M.get_update_ops('local_mth') \
	+ M.get_update_ops('local_nse')\
	+ M.get_update_ops('global_path')\
	+ M.get_update_ops('fusion_node')
	upde = M.get_update_ops('global_path')[:18]
	losscollection = [pix_loss,sym_loss,gen_loss,ip_loss,tv_loss,domain_loss,dis_loss]
	with tf.name_scope('Optimizer'):
		totalLoss = (pix_loss*0.3+sym_loss*0.07+0.001*gen_loss+0.003*ip_loss+0.0001*tv_loss+0.1*domain_loss)
		train_d = tf.train.AdamOptimizer(0.0001).minimize(dis_loss+cls_loss_real,var_list=vard)
		train_g = tf.train.AdamOptimizer(0.0001).minimize(totalLoss,var_list=varg)
		train_e = tf.train.AdamOptimizer(0.0001).minimize(domain_loss)

class data_reader():
	def __init__(self):
		self.filelist = []
		print('Reading data...')
		f = open('trainlist.txt')
		for i in f:
			i = i.strip()
			self.filelist.append(i)
		self.Q = deque()
		for i in range(50):
			print(i)
			self.readbatch(10)
		print('Data length:',len(self.filelist))
		print('Queue length:',len(self.Q))

	def readbatch(self,bsize):
		i = random.sample(self.filelist,bsize)
		for item in i:
			self.Q.append(self.get_sample(item))

	def get_sample(self,i):
		dt = i.split('\t')
		gt = dt[1]
		dt = dt[0]
		clsnum = int(dt[2])
		# prof, le, re, nse, mth, gt
		root_path = '/home/psl/7T/train_data3/'
		faceimg = cv2.imread(root_path+dt+'_face.jpg')
		leimg = cv2.imread(root_path+dt+'_le.jpg')
		reimg = cv2.imread(root_path+dt+'_re.jpg')
		mthimg = cv2.imread(root_path+dt+'_mth.jpg')
		nseimg = cv2.imread(root_path+dt+'_nse.jpg')
		gtimg = cv2.imread(root_path+gt+'_face.jpg')
		faceimg = faceimg.astype(np.float32)
		leimg = leimg.astype(np.float32)
		reimg = reimg.astype(np.float32)
		mthimg = mthimg.astype(np.float32)
		nseimg = nseimg.astype(np.float32)
		gtimg = gtimg.astype(np.float32)
		faceimg = (faceimg/127.5 - 1.0)
		leimg = (leimg/127.5 - 1.0)
		reimg = (reimg/127.5 - 1.0)
		mthimg = (mthimg/127.5 - 1.0)
		nseimg = (nseimg/127.5 - 1.0)
		gtimg = (gtimg/127.5 - 1.0)
		return [faceimg, leimg, reimg, nseimg, mthimg, gtimg, clsnum]

	def get_batch(self,bsize):
		result = []
		for i in range(bsize):
			result.append(self.Q.popleft())
		self.readbatch(bsize)
		return result

def get_data2():
	data2 = []
	f = open('trainlist2.txt')
	for i in f:
		i = i.strip()
		img = cv2.imread(i)
		img = img.astype(np.float32)
		img = (img/127.5 - 1.0)
		img = cv2.resize(img,(122,144))
		M2 = np.float32([[1,0,11],[0,1,0]])
		img = cv2.warpAffine(img,M2,(144,144))
		img = img[8:136,8:136]
		data2.append(img)
	print('data2 length:',len(data2))
	return data2

build_total_graph()
with tf.Session() as sess:
	print('Writing log file...')
	saver = tf.train.Saver()
	reader = data_reader()
	# writer = tf.summary.FileWriter('./logs/',sess.graph)
	# data2 = get_data2()
	M.loadSess('./model/',sess,init=True)
	M.loadSess(modpath='/home/psl/7T/chengyu/modelres/Epoc0Iter20999.ckpt',sess=sess,var_list=M.get_trainable_vars('MainModel'))
	alltensor = [totalLoss,dis_loss,train_d,train_g,updd,updg]
	gentensor = [profHolder,x_fake,gtHolder]
	for iteration in range(100000):
		train_batch = reader.get_batch(BSIZE)
		# p2_batch = random.sample(data2,BSIZE)
		p_batch = [i[0] for i in train_batch]
		l_batch = [i[1] for i in train_batch]
		r_batch = [i[2] for i in train_batch]
		n_batch = [i[3] for i in train_batch]
		m_batch = [i[4] for i in train_batch]
		gt_batch = [i[5] for i in train_batch]
		lb_batch = [i[6] for i in train_batch]
		z_batch = np.random.uniform(size=[BSIZE,ZDIM],low=-1.0,high=1.0)
		feeddict = {profHolder:p_batch, gtHolder:gt_batch, leHolder:l_batch, reHolder:r_batch, mthHolder:m_batch, nseHolder:n_batch, zHolder:z_batch, domainHolder:np.zeros([BSIZE]), clsHolder:lb_batch}
		g_loss, d_loss, _, _, _, _ = sess.run(alltensor, feed_dict=feeddict)
		sess.run([train_g,updg],feed_dict=feeddict)
		feeddict2 = {domainHolder:np.ones([BSIZE]), profHolder:p2_batch}
		sess.run([train_e,upde],feed_dict=feeddict2)
		print('iter:',iteration,'\tgloss:',g_loss,'\tdloss:',d_loss)
		if iteration%10==0:
			losses = sess.run(losscollection,feed_dict=feeddict)
			fout = open('loss.txt','a')
			strout = []
			losses = np.float32(losses).reshape([-1])
			for index in range(len(losses)):
				strout.append(str(losses[index]))
			strout = '\t'.join(strout)+'\n'
			fout.write(strout)
			fout.close()
		if iteration%100==0:
			prof_sample, fake_sample, gt_sample = sess.run(gentensor,feed_dict=feeddict)
			prof_sample = (prof_sample + 1.0) *127.5
			fake_sample = np.clip(fake_sample,-1.0,1.0)
			fake_sample = (fake_sample + 1.0) *127.5
			gt_sample = (gt_sample + 1.0) * 127.5
			prof_sample = prof_sample.astype(np.uint8)
			print(fake_sample.min(),fake_sample.max())
			fake_sample = fake_sample.astype(np.uint8)
			gt_sample = gt_sample.astype(np.uint8)
			for i in range(BSIZE):
				cv2.imwrite('./sample/iter'+str(iteration)+'sample'+str(i)+'prof.jpg',prof_sample[i])
				cv2.imwrite('./sample/iter'+str(iteration)+'sample'+str(i)+'fake.jpg',fake_sample[i])
				cv2.imwrite('./sample/iter'+str(iteration)+'sample'+str(i)+'gndt.jpg',gt_sample[i])
		if iteration%3000==0 and iteration!=0:
			saver.save(sess,'./model/'+str(iteration)+'.ckpt')
import tensorflow as tf 
import model as M 
import numpy as np 
import cv2
from datetime import datetime
import random

ZDIM = 64
IMGPIX = 128
with tf.name_scope('vecinp'):
	z = tf.placeholder(tf.float32,[None,ZDIM])
with tf.name_scope('img'):
	imgholder = tf.placeholder(tf.float32,[None,128,128,1])
with tf.name_scope('classInp'):
	classholder = tf.placeholder(tf.int64,[None])

VARS = {}
BSIZE = 32
LR = 0.0002
BETA=0.4
CLASS = 1000

def gen(inp,shape,reuse=False):
	with tf.variable_scope('Generator',reuse=reuse):
		mod = M.Model(inp,shape)
		mod.fcLayer(4*4*512)
		mod.construct([4,4,512])
		mod.deconvLayer(4,256,stride=2,activation=M.PARAM_RELU,batch_norm=True)#8
		mod.deconvLayer(4,128,stride=2,activation=M.PARAM_RELU,batch_norm=True)#16
		mod.deconvLayer(4,64,stride=2,activation=M.PARAM_RELU,batch_norm=True)#32
		mod.deconvLayer(4,32,stride=2,activation=M.PARAM_RELU,batch_norm=True)#64
		mod.deconvLayer(4,16,stride=2,activation=M.PARAM_RELU,batch_norm=True)#128
		mod.deconvLayer(4,1,stride=1,activation=M.PARAM_TANH,batch_norm=True)
		VARS['g'] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Generator')
		print(len(VARS['g']))
		return mod.get_current_layer()

def dis(inp,shape,reuse=False):
	with tf.variable_scope('Discriminator',reuse=reuse):
		mod = M.Model(inp,shape)
		mod.convLayer(5,16,stride=2,activation=M.PARAM_ELU,batch_norm=True)#64
		mod.convLayer(4,32,stride=2,activation=M.PARAM_ELU,batch_norm=True)#32
		mod.convLayer(4,64,stride=2,activation=M.PARAM_ELU,batch_norm=True)#16
		mod.convLayer(4,128,stride=2,activation=M.PARAM_ELU,batch_norm=True)#8
		mod.convLayer(4,256,stride=2,activation=M.PARAM_ELU,batch_norm=True)#4
		mod.flatten()
		mod.fcLayer(2)
		VARS['d'] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Discriminator')
		print(len(VARS['d']))
		return mod.get_current_layer()

def classifier(inp,shape,reuse=False):
	with tf.variable_scope('Classifier',reuse=reuse):
		mod = M.Model(inp,shape)
		mod.convLayer(5,32,stride=2,activation=M.PARAM_ELU,batch_norm=True)#64
		mod.convLayer(4,64,stride=2,activation=M.PARAM_ELU,batch_norm=True)#32
		mod.convLayer(4,128,stride=2,activation=M.PARAM_ELU,batch_norm=True)#16
		mod.convLayer(4,256,stride=2,activation=M.PARAM_ELU,batch_norm=True)#8
		mod.convLayer(4,512,stride=2,activation=M.PARAM_ELU,batch_norm=True)#4
		mod.flatten()
		mod.fcLayer(ZDIM)
		a = mod.l2norm()
		mod.fcLayer(CLASS)
		VARS['c'] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Classifier')
		print(len(VARS['c']))
		return mod.get_current_layer(),a[0]

generated = gen(z,[None,ZDIM])
disfalse = dis(generated,[None,128,128,1])
distrue = dis(imgholder,[None,128,128,1],reuse=True)
classed,_ = classifier(imgholder,[None,128,128,1])
_,fv = classifier(generated,[None,128,128,1],reuse=True)


with tf.name_scope('lossG'):
	lossG1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.ones([BSIZE],dtype=tf.int64),logits=disfalse))
	lossG2 = tf.reduce_mean(tf.reduce_sum(tf.square(z-fv)))
	lossG3 = tf.reduce_mean((tf.square(imgholder - generated)))
	tf.summary.scalar('lossG1',lossG1)
	tf.summary.scalar('lossG2',lossG2)
	tf.summary.scalar('lossG3',lossG3)
	lossG = lossG1 + lossG2 + lossG3
	# lossG = lossG1
	tf.summary.scalar('lossG',lossG)
with tf.name_scope('lossD'):
	lossD1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.ones([BSIZE],dtype=tf.int64),logits=distrue))
	lossD2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.zeros([BSIZE],dtype=tf.int64),logits=disfalse))
	lossD = 0.5*(lossD1+lossD2)
	tf.summary.scalar('lossD',lossD)
with tf.name_scope('lossC'):
	lossC = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=classholder,logits=classed))
	tf.summary.scalar('lossC',lossC)

with tf.name_scope('opti'):
	with tf.name_scope('optiG'):
		trainG = tf.train.RMSPropOptimizer(LR).minimize(lossG,var_list=VARS['g'])
	with tf.name_scope('optiD'):
		trainD = tf.train.RMSPropOptimizer(LR).minimize(lossD,var_list=VARS['d'])
	with tf.name_scope('iptiC'):
		trainC = tf.train.RMSPropOptimizer(LR).minimize(lossC,var_list=VARS['c'])

# with tf.name_scope('opti'):
# 	with tf.name_scope('optiG'):
# 		trainG = tf.train.AdamOptimizer(learning_rate=LR,beta1=BETA).minimize(lossG,var_list=VARS['g'])
# 	with tf.name_scope('optiD'):
# 		trainD = tf.train.AdamOptimizer(learning_rate=LR,beta1=BETA).minimize(lossD,var_list=VARS['d'])
# 	with tf.name_scope('iptiC'):
# 		trainC = tf.train.AdamOptimizer(learning_rate=LR,beta1=BETA).minimize(lossC,var_list=VARS['c'])

# Use this block when generating imgs
# noise = tf.placeholder(tf.float32,[None,ZDIM])
# _,fv = classifier(imgholder,[None,128,128,1])
# generated = gen(fv+noise,[None,ZDIM])

def getGeneratedImg(sess,it):
	a = np.random.uniform(size=[4,ZDIM],low=-1.0,high=1.0)
	a = a/np.linalg.norm(a,axis=1,keepdims=True)
	img = sess.run(generated,feed_dict={z:a})
	img = img+1
	img = img*127
	img = img.astype(np.uint8)
	for i in range(4):
		cv2.imwrite('res/iter'+str(it)+'img'+str(i)+'.jpg',cv2.resize(img[i],(128,128)))

def getData():
	f = open('avclb2.txt')
	dt = []
	counter = 0
	for line in f:
		counter+=1
		if (counter+1)%1000==0:
			print(counter+1)
			# break
		l = line.replace('\n','').split(' ')
		img = np.float32(cv2.resize(cv2.imread(l[0],0),(IMGPIX,IMGPIX))).reshape([128,128,1])
		img = img / 127.5
		img = img -1
		lb = int(l[1])
		dt.append((img,lb))
	return dt

def training():
	merged = tf.summary.merge_all()
	data = getData()
	saver = tf.train.Saver()
	with tf.Session() as sess:
		writer = tf.summary.FileWriter('./logs/',sess.graph)
		M.loadSess('./model/',sess=sess)
		counter = M.counter
		for i in range(1000000):
			counter+=1
			sample = random.sample(data,BSIZE)
			x_train = [i[0] for i in sample]
			y_train = [i[1] for i in sample]
			a = np.random.uniform(size=[BSIZE,ZDIM],low=-1.0,high=1.0)
			a = a/np.linalg.norm(a,axis=1,keepdims=True)
			# ge = sess.run(generated,feed_dict={z:a})
			for _ in range(5):
				sess.run(trainG,feed_dict={z:a,imgholder:x_train})
			_,_,mg,lsd,lsg,lsc = sess.run([trainC,trainD,merged,lossD,lossG,lossC],feed_dict={z:a,imgholder:x_train,classholder:y_train})
			if (i)%5 == 0:
				writer.add_summary(mg,counter)
				print('iter:',i)
				print('lsd:',lsd)
				print('lsg:',lsg)
				print('lsc:',lsc)
			if (i+1)%100==0:
				getGeneratedImg(sess,i+1)	
			if (i+1)%1000==0:
				saver.save(sess,'./model/ModelCounter'+str(counter)+'.ckpt')

def getSample():
	with tf.Session() as sess:
		data = getData()
		M.loadSess('./model/',sess=sess)
		for i in range(20):
			x_train = random.sample(data,1)
			# print(x_train[0].shape)
			x_train = np.float32(x_train[0][0]).reshape([-1,128,128,1])
			for j in range(8):
				# a = np.random.uniform(size=[1,ZDIM],low=-0.2,high=0.2)
				a = np.zeros([1,ZDIM],dtype=np.float32)
				genimg = sess.run(generated,feed_dict={imgholder:x_train,noise:a})
				genimg = (genimg+1)*127
				genimg = genimg.astype(np.uint8)
				cv2.imwrite('./sampleimg/'+str(i)+'gen'+str(j)+'.jpg',cv2.resize(genimg[0],(128,128)))
				cv2.imwrite('./sampleimg/'+str(i)+'org.jpg',cv2.resize(((x_train[0]+1)*127).astype(np.uint8),(128,128)))

# getSample()
training()
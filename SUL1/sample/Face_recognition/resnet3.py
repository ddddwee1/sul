import tensorflow as tf 
import numpy as np 
import model as M 

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

def res_18():
	with tf.name_scope('imginput'):
		imgholder = tf.placeholder(tf.float32,[None,128,128,3])
	mod = M.Model(imgholder,[None,128,128,3])
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
	return imgholder,featurelayer

with tf.variable_scope('MainModel'):
	imgholder,featurelayer = res_18()

saver = tf.train.Saver()
sess = tf.Session()
M.loadSess('./model/',sess=sess)
saver.save(sess,'./model.ckpt')

def eval(img):
	res = sess.run(featurelayer,feed_dict={imgholder:img})
	return res

def __exit__():
	sess.close()
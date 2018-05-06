import numpy as np 
import tensorflow as tf 
import model as M 

def build_model(inp):
	with tf.variable_scope('CapsNet'):
		mod = M.Model(inp)
		mod.reshape([-1,28,28,1])
		mod.convLayer(5,4*16,stride=2,activation=M.PARAM_RELU)
		mod.capsulization(4,16)
		mod.caps_conv(3,8,32,stride=2)
		mod.caps_conv(3,8,32,stride=2)
		mod.caps_flatten()
		mod.capsLayer(10,16,3,BSIZE)
		mod.capsDown()
	return mod.get_current_layer()

def build_graph():
	inpholder = tf.placeholder(tf.float32,[None,28*28])
	labholder = tf.placeholder(tf.float32,[None,10])

	out = build_model(inpholder)
	with tf.variable_scope('length'):
		length = tf.sqrt(tf.reduce_sum(tf.square(out),-1))

	with tf.variable_scope('marg_loss'):
		marg_loss = labholder * tf.square(tf.maximum(0.,0.9-length)) + 0.5* (1-labholder) * tf.square(tf.maximum(0.,length-0.1))
		marg_loss = tf.reduce_mean(tf.reduce_sum(marg_loss,1))

	with tf.variable_scope('opti'):
		train_step = tf.train.AdamOptimizer(0.0001).minimize(marg_loss)

	with tf.variable_scope('Accuracy'):
		acc = M.accuracy(length,tf.argmax(labholder,1))

	return inpholder,labholder,marg_loss,train_step,acc

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
BSIZE = 100
MAX_ITER = 20000

inph,lbh,mgls,ts,acc = build_graph()
tensors = [ts,mgls,acc]

with tf.Session() as sess:
	tf.summary.FileWriter('./logs/',sess.graph)
	saver = tf.train.Saver()
	M.loadSess('./model/',sess,init=True)
	for it in range(MAX_ITER):
		x_train, y_train = mnist.train.next_batch(BSIZE)
		_,ls,ac = sess.run(tensors,feed_dict={inph:x_train,lbh:y_train})
		if it%10==0:
			print('Iter:%d\tLoss:%.5f\tAccuracy:%.4f'%(it,ls,ac))
		if it % 100 == 0:
			x_dt = mnist.test.images
			y_dt = mnist.test.labels
			acttl = 0
			for i in range(100):
				x_train = x_dt[i*100:i*100+100]
				y_train = y_dt[i*100:i*100+100]
				ac = sess.run(acc,feed_dict={inph:x_train, lbh:y_train})
				acttl += ac
			print('Test acc:',acttl/100)
		if it%500==0 and it>0:
			saver.save(sess,'./model/%d.ckpt'%it)
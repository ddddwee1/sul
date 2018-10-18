import tensorflow as tf 
import model as M 
import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
BSIZE = 128

def build_model(img_input):
	mod = M.Model(img_input,[None,28*28])
	mod.reshape([-1,28,28,1])
	mod.convLayer(5,16,activation=M.PARAM_LRELU)
	mod.maxpoolLayer(2)
	mod.convLayer(5,16,activation=M.PARAM_LRELU)
	mod.maxpoolLayer(2)
	mod.convLayer(5,16,activation=M.PARAM_LRELU)
	mod.maxpoolLayer(2)
	mod.flatten()
	mod.fcLayer(50,activation=M.PARAM_TANH)
	mod.fcLayer(50,activation=M.PARAM_TANH)
	mod.fcLayer(10)
	return mod.get_current_layer()

def build_graph():
	img_holder = tf.placeholder(tf.float32,[None,28*28])
	lab_holder = tf.placeholder(tf.float32,[None,10])
	last_layer = build_model(img_holder)
	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=lab_holder,logits=last_layer))
	accuracy = M.accuracy(last_layer,tf.argmax(lab_holder,1))
	train_step = M.Trainer(0.0001,loss).train()
	return img_holder,lab_holder,loss,train_step,accuracy,last_layer

img_holder,lab_holder,loss,train_step,accuracy,last_layer = build_graph()

with tf.Session() as sess:
	saver = tf.train.Saver()
	M.loadSess('./model/',sess,init=True)
	for i in range(1000):
		x_train, y_train = mnist.train.next_batch(BSIZE)
		_,acc,ls = sess.run([train_step,accuracy,loss],feed_dict={img_holder:x_train,lab_holder:y_train})
		if i%100==0:
			print('iter',i,'\t|acc:',acc,'\tloss:',ls)
		if i%1000==0:
			acc = sess.run(accuracy,feed_dict={img_holder:mnist.test.images, lab_holder:mnist.test.labels})
			print('Test accuracy:',acc)
	saver.save(sess,'./model/abc.ckpt')


# -------------------------
# Run time testing
# -------------------------


# import time
# with tf.Session() as sess:
# 	M.loadSess('./model/',sess,init=True)
# 	for i in range(1500):
# 		x = np.ones([9,28*28]).astype(np.float32)
# 		sess.run(last_layer,feed_dict={img_holder:x})
# 		if i==500:
# 			timea = time.time()
# 	timeb = time.time()
# 	print('Time elapsed:', timeb - timea)
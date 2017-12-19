import tensorflow as tf 
import model as M 
import numpy as np 


# ------build main model and decoder -----------

def build_model(inph):
	with tf.variable_scope('CapsModel'):
		mod = M.Model(inph,[None,28*28])
		mod.reshape([-1,28,28,1])
		mod.convLayer(11,256,activation=M.PARAM_RELU,pad='VALID')
		mod.primaryCaps(9,8,32,stride=2,activation=M.PARAM_RELU,pad='VALID')
		mod.capsLayer(10,16,3,BSIZE)
		mod.capsDown()
	return mod.get_current_layer()

def build_decoder(inp,labh):
	with tf.variable_scope('Decoder'):
		mod = M.Model(inp,[None,10,16])
		mod.capsMask(labh)
		mod.fcLayer(512,activation=M.PARAM_RELU)
		mod.fcLayer(1024,activation=M.PARAM_RELU)
		mod.fcLayer(28*28,activation=M.PARAM_SIGMOID)
	return mod.get_current_layer()


# -------define margin loss and reconstruction loss ---------
def build_graph():
	inp_holder = tf.placeholder(tf.float32,[None,28*28])
	label_holder = tf.placeholder(tf.float32,[None,10])

	out = build_model(inp_holder)
	decoded = build_decoder(out,label_holder)

	with tf.variable_scope('Length'):
		length = tf.sqrt(tf.reduce_sum(tf.square(out),-1))

	with tf.variable_scope('Loss'):
		marg_loss = label_holder * tf.square(tf.maximum(0.,0.9-length)) + 0.5 * (1-label_holder) * tf.square(tf.maximum(0.,length-0.1))
		marg_loss = tf.reduce_mean(tf.reduce_sum(marg_loss,1),0)
		recon_loss = tf.reduce_mean(tf.square(inp_holder-decoded))

	with tf.variable_scope('Opti'):
		train_step = tf.train.AdamOptimizer(0.0001).minimize(marg_loss+0.0005*784*recon_loss) # multiplier from MSE to SE

	with tf.variable_scope('Accuracy'):
		accuracy = M.accuracy(length,tf.argmax(label_holder,1))

	return inp_holder,label_holder,recon_loss,marg_loss,train_step,accuracy


# ------training-----
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
BSIZE = 100
MAX_ITER = 20000

inph,labh,recon_loss,marg_loss,train_step,accuracy = build_graph()
tensors = [train_step,recon_loss,marg_loss,accuracy]

with tf.Session() as sess:
	tf.summary.FileWriter('./logs/',sess.graph)
	saver = tf.train.Saver()
	M.loadSess('./model/',sess,init=True)
	for iteration in range(MAX_ITER):
		x_train, y_train = mnist.train.next_batch(BSIZE)
		_,lossr,lossm,ac = sess.run([train_step,recon_loss,marg_loss,accuracy], feed_dict={inph:x_train, labh:y_train})
		if iteration % 10 ==0:
			print('Iter:',iteration,'\tLossr:',lossr,'\tLossm:',lossm,'\tAcc:',ac)
		if iteration % 100 == 0:
			x_dt = mnist.test.images
			y_dt = mnist.test.labels
			acttl = 0
			for i in range(100):
				x_train = x_dt[i*100:i*100+100]
				y_train = y_dt[i*100:i*100+100]
				ac = sess.run(accuracy,feed_dict={inph:x_train, labh:y_train})
				acttl += ac
			print('Test acc:',acttl/100)
		if iteration % 400 == 0:
			saver.save(sess,'./model/caps_model_iter%d.ckpt'%(iteration))
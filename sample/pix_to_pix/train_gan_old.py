import tensorflow as tf 
import model as M 
import numpy as np 
import cv2 
import random

def gen(inp,reuse=False,train=True):
	with tf.variable_scope('gen',reuse=reuse):
		mod = M.Model(inp,[None,256,256,3])
		mod.set_bn_training(train)
		conv0 = mod.convLayer(3,64,activation=M.PARAM_LRELU,batch_norm=True)
		conv1 = mod.convLayer(4,128,stride=2,activation=M.PARAM_LRELU,batch_norm=True)#128
		conv2 = mod.convLayer(4,256,stride=2,activation=M.PARAM_LRELU,batch_norm=True)#64
		conv3 = mod.convLayer(4,512,stride=2,activation=M.PARAM_LRELU,batch_norm=True)#32
		conv4 = mod.convLayer(4,512,stride=2,activation=M.PARAM_LRELU,batch_norm=True)#16
		conv5 = mod.convLayer(4,512,stride=2,activation=M.PARAM_LRELU,batch_norm=True)#8
		conv6 = mod.convLayer(4,512,stride=2,activation=M.PARAM_LRELU,batch_norm=True)#4
		conv7 = mod.convLayer(4,512,stride=2,activation=M.PARAM_LRELU,batch_norm=True)#2
		deconv7 = mod.deconvLayer(4,512,stride=2,activation=M.PARAM_RELU,batch_norm=True)#4
		mod.dropout(0.5)
		mod.concat_to_current(conv6)
		deconv6 = mod.deconvLayer(4,512,stride=2,activation=M.PARAM_RELU,batch_norm=True)#8
		mod.dropout(0.5)
		mod.concat_to_current(conv5)
		deconv5 = mod.deconvLayer(4,512,stride=2,activation=M.PARAM_RELU,batch_norm=True)#16
		mod.dropout(0.5)
		mod.concat_to_current(conv4)
		deconv4 = mod.deconvLayer(4,512,stride=2,activation=M.PARAM_RELU,batch_norm=True)#32
		mod.concat_to_current(conv3)
		deconv3 = mod.deconvLayer(4,512,stride=2,activation=M.PARAM_RELU,batch_norm=True)#64
		mod.concat_to_current(conv2)
		deconv2 = mod.deconvLayer(4,512,stride=2,activation=M.PARAM_RELU,batch_norm=True)#128
		mod.concat_to_current(conv1)
		deconv1 = mod.deconvLayer(4,512,stride=2,activation=M.PARAM_RELU,batch_norm=True)#256
		mod.concat_to_current(conv0)
		outlayer = mod.convLayer(3,1,activation=M.PARAM_TANH,batch_norm=True)
		return outlayer[0]

def dis(inp1,inp2,reuse=False,train=True):
	with tf.variable_scope('dis',reuse=reuse):
		mod = M.Model(inp1,[None,256,256,1])
		mod.set_bn_training(train)
		conv0a = mod.convLayer(4,32,stride=2,activation=M.PARAM_LRELU,batch_norm=False)
		mod.set_current([inp2,[None,256,256,3]])
		conv0b = mod.convLayer(4,32,stride=2,activation=M.PARAM_LRELU,batch_norm=False)#128
		mod.concat_to_current(conv0a)
		conv1 = mod.convLayer(4,128,stride=2,activation=M.PARAM_LRELU,batch_norm=True)#64
		conv2 = mod.convLayer(4,256,stride=2,activation=M.PARAM_LRELU,batch_norm=True)#32
		conv3 = mod.convLayer(4,512,stride=2,activation=M.PARAM_LRELU,batch_norm=True)#16
		conv4 = mod.convLayer(3,1)#16
		return conv4[0]

def build_graph(train=True):
	with tf.name_scope('inp'):
		imgholder = tf.placeholder(tf.float32,[None,256,256,3])
	with tf.name_scope('gnd'):
		gndholder = tf.placeholder(tf.float32,[None,256,256,1])

	x_fake = gen(imgholder,train=train)
	d_fake = dis(x_fake,imgholder,train=train)
	d_real = dis(gndholder,imgholder,reuse=True,train=train)

	g_loss_L1 = tf.reduce_mean(tf.abs(x_fake - gndholder))
	g_loss_lg = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake,labels=tf.ones_like(d_fake)))

	d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real,labels=tf.ones_like(d_real)))
	d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake,labels=tf.zeros_like(d_fake)))

	varg = M.get_trainable_vars('gen')
	vard = M.get_trainable_vars('dis')
	updg = M.get_update_ops('gen')
	updd = M.get_update_ops('dis')

	with tf.name_scope('optimizer'):
		train_d = tf.train.AdamOptimizer(0.001,beta1=0.5).minimize(d_loss_real+d_loss_fake, var_list=vard)
		train_g = tf.train.AdamOptimizer(0.001,beta1=0.5).minimize(g_loss_lg + g_loss_L1*10, var_list=varg)

	return [imgholder,gndholder],[g_loss_L1+g_loss_lg,d_loss_fake+d_loss_real],[updg,updd],[train_g,train_d],x_fake,[g_loss_lg,d_loss_fake+d_loss_real]

def read_data():
	f = open('train_list.txt')
	print('Reading data...')
	data = []
	for i in f:
		i = i.strip()
		img1 = cv2.imread(i.replace('.jpg','.tif'),0)
		# print(i)
		img2 = cv2.imread(i.replace('.jpg','_of.jpg'))
		img1 = cv2.resize(img1,(256,256))
		img2 = cv2.resize(img2,(256,256))
		img1 = img1.astype(np.float32)
		img2 = img2.astype(np.float32)
		img1 = img1/127.5-1
		img2 = img2/127.5-1
		img1 = img1.reshape([256,256,1])
		data.append([img2,img1])
	return data

def show_sample(gen,img,gnd,iter):
	for i in range(len(gen)):
		gen0 = gen[i]
		img0 = img[i]
		gnd0 = gnd[i]
		gen0 = (gen0+1.0)*127.5
		img0 = (img0+1.0)*127.5
		gnd0 = (gnd0+1.0)*127.5
		gen0 = gen0.astype(np.uint8).reshape([256,256])
		img0 = img0.astype(np.uint8)
		gnd0 = gnd0.astype(np.uint8)
		cv2.imwrite('./sample/'+str(iter)+'_'+str(i)+'_gen.jpg',gen0)	
		cv2.imwrite('./sample/'+str(iter)+'_'+str(i)+'_img.jpg',img0)
		cv2.imwrite('./sample/'+str(iter)+'_'+str(i)+'_gnd.jpg',gnd0)

BSIZE = 4

def main():
	holders, losses, upops, trains,x_fake,lb_ls = build_graph()
	saver = tf.train.Saver()
	with tf.Session() as sess:
		M.loadSess('./model/',sess,init=True)
		data = read_data()
		for iteration in range(10000):
			train_batch = random.sample(data,BSIZE)
			img_batch = [i[0] for i in train_batch]
			gnd_batch = [i[1] for i in train_batch]
			feeddict = {holders[0]: img_batch, holders[1]:gnd_batch}
			g_loss,d_loss,_,_,_,_,lb_g,lb_d = sess.run(losses+upops+trains+lb_ls, feed_dict=feeddict)
			print('Iter:',iteration,'\tLoss_g:',g_loss,'\tLb_g:',lb_g,'\tLoss_d:',d_loss)
			while lb_g>lb_d+0.7:
				train_batch = random.sample(data,BSIZE)
				img_batch = [i[0] for i in train_batch]
				gnd_batch = [i[1] for i in train_batch]
				feeddict = {holders[0]: img_batch, holders[1]:gnd_batch}
				g_loss,d_loss,_,_,lb_g,lb_d = sess.run(losses+[upops[0],trains[0]]+lb_ls,feed_dict=feeddict)
				print('Iter:',iteration,'\tLoss_g:',g_loss,'\tLb_g:',lb_g,'\tLoss_d:',d_loss)
			if iteration%100==0 and iteration>0:
				saver.save(sess,'./model/'+str(iteration)+'.ckpt')
			if iteration%20==0:
				gen = sess.run(x_fake,feed_dict=feeddict)
				show_sample(gen,img_batch,gnd_batch,iteration)

def test_samples(fname):
	img_batch = []
	for i in fname:
		img = cv2.resize(cv2.imread(i),(256,256)).reshape([256,256,3])
		img = np.float32(img)
		img = img/127.5 - 1.0
		img_batch.append(img)
	holders, losses, upops, trains,x_fake,lb_ls = build_graph(train=False)
	with tf.Session() as sess:
		M.loadSess('./model/',sess)
		gen = sess.run(x_fake,feed_dict={holders[0]:img_batch})
		show_sample(gen,img_batch,img_batch,0)

main()

# test_samples(['007_of.jpg','091_of.jpg','153_of.jpg'])
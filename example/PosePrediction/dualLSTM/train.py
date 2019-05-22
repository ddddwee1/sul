# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ''
import data_utils
import numpy as np 
import pickle 
import config 
import util 
import network
import model3 as M 
import tensorflow as tf 

def grad_loss(x, model):
	batch_inp, batch_gt = x 
	with tf.GradientTape() as tape:
		predictions = model(batch_inp, len(batch_gt))
		loss = tf.reduce_mean([tf.square(p,gt) for p,gt in zip(predictions, list(batch_gt))])
	grad = tape.gradient(loss, model.trainable_variables)
	return grad, loss

BATCH_SIZE = config.BATCH_SIZE
SEQ_IN = config.SEQ_IN
SEQ_OUT = config.SEQ_OUT
IN_DIM = config.IN_DIM
ACTIONS = config.ACTIONS 
MAX_ITERS = config.MAX_ITERS

data = pickle.load(open('data.pkl','rb'))
train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = data 

with tf.device('/:cpu'):
	net = network.PosePredNet()
	saver = M.Saver(net)
	saver.restore('./model/')
	optim = tf.optimizers.Adam(0.001)

	for i in range(MAX_ITERS+1):
		batch_enc_inp, batch_dec_inp, batch_gt = util.get_batch(train_set, False, ACTIONS)
		batch_inp = np.concatenate([batch_enc_inp, batch_dec_inp[:,:1]], axis=1)
		batch_inp = np.transpose(batch_inp, [1,0,2]).astype(np.float32)
		batch_gt = np.transpose(batch_gt, [1,0,2]).astype(np.float32)
		
		grad, loss = grad_loss([batch_inp, batch_gt], net)
		optim.apply_gradients(zip(grad, net.trainable_variables))

		if i%10==0:
			print('Iter:%d\tLoss:%.4f'%(i, loss))
		
		if i%5000==0 and i>0:
			saver.save('./model/model.ckpt')

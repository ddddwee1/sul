import tensorflow as tf 
import model3 as M 
import datareader 
import numpy as np 
import tqdm 
import network

def grad_loss(x, model):
	x2d, x3d = x
	with tf.GradientTape() as tape:
		pred, K, reprojected, crit_fake = model(x2d)
		crit_real = model.crit(x3d)

		crit_dis = tf.reduce_mean(tf.square(crit_real - tf.ones_like(crit_real))) + tf.reduce_mean(tf.square(crit_fake - tf.zeros_like(crit_fake)))
		crit_gen = tf.reduce_mean(tf.square(crit_fake - tf.ones_like(crit_fake)))

		rep_loss = tf.reduce_mean(tf.square(pred - x2d))

		KK = tf.matmul(K, K, transpose_b=True)
		K_trace = tf.expand_dims(tf.expand_dims(tf.trace(KK), -1), -1)
		K_loss = tf.reduce_mean(tf.abs(KK / K_trace - tf.eye(2))) 

		loss_total_gen = crit_gen + rep_loss + K_loss

	gen_var = model.get_gen_vars()
	dis_var = model.dis.trainable_variables
	grads = tape.gradient([loss_total_gen, crit_dis], [gen_var, dis_var])
	return grads, [crit_dis, crit_gen, rep_loss, K_loss]

reader = datareader.DataReader(16)
model = network.RepNet()
optim = tf.optimizers.Adam(0.0001, 0.5)
saver = M.Saver(model)
saver.restore('./model/')

MAXITER = 10000

bar = tqdm(range(MAXITER+1))
for i in bar:
	batch = reader.get_next()
	grads, lss = grad_loss(batch, model)

	gen_var = model.get_gen_vars()
	dis_var = model.dis.trainable_variables
	optim.apply_gradients(zip(grads[0], gen_var))
	optim.apply_gradients(zip(grads[1], dis_var))

	bar.set_description('CDis:%.4f CGen:%.4f Rep:%.4f K:%.4f'%(lss[0], lss[1], lss[2], lss[3]))

	if i%1000==0 and i>0:
		saver.save('./model/repnet.ckpt')

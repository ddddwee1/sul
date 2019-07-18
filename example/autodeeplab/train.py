import model3 as M 
import tensorflow as tf 
import numpy as np 
import network
import datareader

BSIZE = 128*4
EPOCH = 30
# data_reader = datareader.DataReader('imglist_iccv.txt', BSIZE)
tf.keras.backend.set_learning_phase(True)

def loss_func(out, label):
	lb = tf.convert_to_tensor(label)
	out = tf.nn.log_softmax(out, -1)
	loss = tf.reduce_sum(tf.reduce_sum(- out * lb , -1)) / BSIZE
	return loss 

def grad_loss(x, model):
	data, label = x
	with tf.gradient_tape() as tape:
		out = model(data, label)
		loss = loss_func(out, label)
	acc = M.accuracy(out, label, one_hot=False)
	grads = tape.gradient(loss, model.trainable_variables)
	return grads, [loss, acc]

def lr_decay(step):
	lr = 0.1
	step = step/20000
	step = tf.math.floor(step)
	step = tf.math.pow(0.1, step)
	lr = lr * step 
	return lr 

model = network.FaceRecogNet(512,100)

_ = model(np.float32(np.ones([1,128,128,3])), np.float32(np.eye(100)[0]))
vs = model.trainable_variables
for v in vs:
	print(v.name)

# t0 = time.time()
# LR = M.LRScheduler(lr_decay)
# print('Label number:', data_reader.max_label+1)
# model = network.FaceRecogNet(512, data_reader.max_label + 1)
# optimizer = tf.optimizers.SGD(LR, 0.9)

# saver = M.Saver(model)
# # saver.restore('./model/')

# _ = model(np.float32(np.ones([1,112,112,3])), np.float32(np.eye(data_reader.max_label+1)[0]))

# accmeter = M.EMAMeter(0.9)
# lossmeter = M.EMAMeter(0.9)
# lsttlmeter = M.EMAMeter(0.9)

# for ep in range(EPOCH):
# 	for it in range(data_reader.iter_per_epoch):
# 		batch = data_reader.get_next()
# 		grad, lsacc = grad_loss(batch, model)
# 		optimizer.apply_gradients(zip(grad, model.trainable_variables))

# 		if it%10==0:
# 			t1 = time.time()
# 			img_sec = 10 * BSIZE / (t1-t0)
# 			lsttl = lsacc[0]
# 			ls = lsacc[1]
# 			acc = lsacc[2]
# 			lsttl = lsttlmeter.update(lsttl.numpy())
# 			ls = lossmeter.update(ls.numpy())
# 			acc = accmeter.update(acc.numpy())
# 			t0 = t1 
# 			print('Epoch:%d\tIter:%d\tLoss0:%.6f\tLoss:%.6f\tAcc:%.6f\tSpeed:%.2f'%(ep, it, lsttl, ls, acc, img_sec))

# 		if it%5000==0 and it>0:
# 			saver.save('./model/%d_%d.ckpt'%(ep,it))
# 			t0 = time.time()
# 	saver.save('./model/%d_%d.ckpt'%(ep,it))
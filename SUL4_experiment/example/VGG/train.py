import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import tensorflow as tf 
import SUL.Model as M 
import SUL.util as util 
import SUL.parallel
import datareader
import numpy as np 
import time 
import network 

class FaceVGG(M.Model):
	def initialize(self, num_classes):
		self.vgg = network.VGG19()
		self.classifier = M.MarginalCosineLayer(num_classes)

	def forward(self, x, label):
		feat = self.vgg(x)
		logits = self.classifier(feat, label, 1.0, 0.0, 0.0)
		logits = logits * 64
		return logits

BSIZE = 256*4
EPOCH = 30
data_reader = datareader.get_datareader('imglist_iccv_small.txt', BSIZE, processes=16, gpus=4)
tf.keras.backend.set_learning_phase(True)

def lr_decay(step):
	lr = 0.01
	step = step/10000
	step = tf.math.floor(step)
	step = tf.math.pow(0.1, step)
	lr = lr * step 
	return lr 

def grad_loss(x, model):
	data, label = x
	with tf.GradientTape() as tape:
		data = (data / 127.5) - 1.
		out = model(data, label)
		wd = M.weight_decay_v2(0.0005, model)
		# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=label))
		lb = tf.convert_to_tensor(label)
		out = tf.nn.log_softmax(out, -1)
		loss = tf.reduce_sum(tf.reduce_sum(- out * lb , -1)) / BSIZE
		loss_ttl = loss + wd
	acc = M.accuracy(out, label, one_hot=False)
	grads = tape.gradient(loss_ttl, model.trainable_variables)
	return grads, [loss_ttl, loss, acc]

# monitoring time
t0 = time.time()
LR = util.LRScheduler(lr_decay)

with tf.device('/cpu:0'):
	model = FaceVGG(data_reader.max_label + 1)
	optimizer = tf.optimizers.SGD(LR, 0.9)
	saver = M.Saver(model)
	saver.restore('./model/')

	_ = model(np.float32(np.ones([1,128,128,3])), np.float32(np.eye(data_reader.max_label+1)[0]))
	
	pt = SUL.parallel.ParallelTraining(model, optimizer, [0,1,2,3], grad_loss_fn=grad_loss) 

	accmeter = util.EMAMeter(0.9)
	lossmeter = util.EMAMeter(0.9)
	lsttlmeter = util.EMAMeter(0.9)
	summary_writer = util.Summary('log.json', save_interval=100)

	for ep in range(EPOCH):
		for it in range(data_reader.iter_per_epoch):
			batch = data_reader.get_next()
			lsacc = pt.train_step(batch)

			lsttl = tf.reduce_sum([_[0] for _ in lsacc])
			ls = tf.reduce_sum([_[1] for _ in lsacc])
			acc = tf.reduce_mean([_[2] for _ in lsacc])

			summary_writer.push('loss', float(ls.numpy()))
			summary_writer.push('acc', float(acc.numpy()))
			summary_writer.push('loss_ttl', float(lsttl.numpy()))
			summary_writer.step()

			lsttl = lsttlmeter.update(lsttl.numpy())
			ls = lossmeter.update(ls.numpy())
			acc = accmeter.update(acc.numpy())

			if it%10==0:
				t1 = time.time()
				img_sec = 10*BSIZE / (t1-t0)
				t0 = t1 
				lr = lr_decay(optimizer.iterations)
				print('Epoch:%d\tIter:%d\tLoss0:%.6f\tLoss:%.6f\tAcc:%.6f\tSpeed:%.2f\tLR:%f'%(ep, it, lsttl, ls, acc, img_sec, lr))

			if it%1900==0 and it>0:
				saver.save('./model/%d_%d.ckpt'%(ep,it))
				t0 = time.time()
		saver.save('./model/%d_%d.ckpt'%(ep,it))

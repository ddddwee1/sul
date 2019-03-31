import tensorflow as tf 
import model3 as M 
import numpy as np 
import resnet
import datareader
import time 

class FaceResNet(M.Model):
	def initialize(self, num_classes):
		self.resnet = resnet.ResNet([64,64,128,256,512], [3, 4, 14, 3], 512)
		self.classifier = M.Dense(num_classes)

	def forward(self, x):
		feat = self.resnet(x)
		logits = self.classifier(feat)
		return logits

BSIZE = 320
EPOCH = 30
data_reader = datareader.DataReader('img_list.txt', BSIZE)
tf.keras.backend.set_learning_phase(True)

with tf.device('/cpu:0'):
	model = FaceResNet(data_reader.max_label + 1)
	optimizer = tf.keras.optimizers.Adam(0.0001)


def grad_loss(x):
	data, label = x
	with tf.GradientTape() as tape:
		out = model(data)
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=label))
		acc = M.accuracy(out, label, one_hot=False)
	grads = tape.gradient(loss, model.trainable_variables)
	return grads, [loss, acc]

@tf.function
def train_step(data):
	rr = []
	for idx,i in enumerate([0,1,2,3]):
		with tf.device('/gpu:%d'%i):
			rr.append(grad_loss(data[idx]))
			print('ModelGPU%d'%i)
	losses = []
	grads = [i[0] for i in rr]
	grads = [sum(g) for g in zip(*grads)]
	for i in rr:
		losses.append(i[1])
	optimizer.apply_gradients(zip(grads, model.trainable_variables))
	return losses


# monitoring time
t0 = time.time()
# batch = data_reader.get_next()
with tf.device('/cpu:0'):
	
	saver = M.Saver(model)
	pt = M.ParallelTraining(model, optimizer, [0,1,2,3]) 

	for ep in range(EPOCH):
		for it in range(data_reader.iter_per_epoch):
			batch = data_reader.get_next()
			batch_distribute = pt.split_data(batch)
			tt1 = time.time()
			# rr = pt.compute_grad_loss(data=batch_distribute, grad_loss_fn=grad_loss)
			# tt2 = time.time()
			# print(tt2-tt1)
			# grads, lsacc = pt.process_rr(rr)
			# tt3 = time.time()
			# print(tt3-tt2)
			# pt.apply_grad(grads)
			lsacc = train_step(batch_distribute)
			tt4 = time.time()
			print(tt4-tt1)

			if it%10==0:
				t1 = time.time()
				img_sec = 10 * BSIZE / (t1-t0)
				ls = tf.reduce_mean([_[0] for _ in lsacc])
				acc = tf.reduce_mean([_[1] for _ in lsacc])
				t0 = t1 
				print('Epoch:%d\tIter:%d\tLoss:%.6f\tAcc:%.6f\tSpeed:%.2f'%(ep, it, ls, acc, img_sec))

			if it%5000==0 and it>0:
				saver.save('./model/%d_%d.h5')
				t0 = time.time()

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

def grad_loss(x, model):
	data, label = x
	with tf.gradientTape() as tape:
		out = model(data)
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=label))
		acc = M.accuracy(out, label, one_hot=False)
	grads = tape.gradient(loss, model.trainable_variables)
	return grads, [loss, acc]

BSIZE = 512
EPOCH = 30
data_reader = datareader.DataReader('imglist.txt', BSIZE)

# monitoring time
t0 = time.time()

with tf.device('/cpu:0'):
	model = FaceResNet(data_reader.max_label + 1)
	optimizer = tf.keras.optimizers.Adam(0.0001)
	saver = M.Saver(model)
	pt = M.ParallelTraining(model, optimizer, [0,1,2,3]) 

	for ep in range(EPOCH):
		for it in range(data_reader.iter_per_epoch):
			batch = reader.get_next()
			batch_distribute = pt.split_data(batch)
			grads, lsacc = pt.compute_grad_loss(data=batch_distribute, grad_loss_fn=grad_loss)
			pt.apply_grad()

			if it%10==0:
				t1 = time.time()
				img_sec = 10 * BSIZE / (t1-t0)
				t0 = t1 
				print('Epoch:%d\tIter:%d\tLoss:%.6f\tAcc:%.6f\tSpeed:%.2f'%(ep, it, lsacc[0], lsacc[1], img_sec))

			if it%5000==0 and it>0:
				saver.save('./model/%d_%d.h5')
				t0 = time.time()

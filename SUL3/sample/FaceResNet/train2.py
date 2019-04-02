import tensorflow as tf 
import model3 as M 
import numpy as np 
import resnet
import losspart
import datareader
import time 

class FaceResNet(M.Model):
	def initialize(self, num_classes):
		self.resnet = resnet.ResNet([64,64,128,256,512], [3, 4, 14, 3], 512)
		self.classifier = losspart.ArcFace(num_classes)

	def forward(self, x):
		feat = self.resnet(x)
		feat = tf.nn.dropout(feat, 0.6)
		logits = self.classifier(feat)
		logits = logits * 40
		return logits

BSIZE = 320
EPOCH = 30
data_reader = datareader.DataReader('img_list.txt', BSIZE)
tf.keras.backend.set_learning_phase(True)

def grad_loss(x, model):
	data, label = x
	with tf.GradientTape() as tape:
		out = model(data)
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=label))
	acc = M.accuracy(out, label, one_hot=False)
	grads = tape.gradient(loss, model.trainable_variables)
	return grads, [loss, acc]

# monitoring time
t0 = time.time()
# batch = data_reader.get_next()
with tf.device('/cpu:0'):
	model = FaceResNet(data_reader.max_label + 1)
	optimizer = tf.optimizers.Adam(0.0001)
	saver = M.Saver(model, optimizer)
	saver.restore('./model/')
	
	pt = M.ParallelTraining(model, optimizer, [0,1,2,3], grad_loss_fn=grad_loss, input_size=[112,112,3]) 

	for ep in range(EPOCH):
		for it in range(data_reader.iter_per_epoch):
			batch = data_reader.get_next()
			batch_distribute = pt.split_data(batch)
			lsacc = pt.train_step(batch_distribute)

			if it%10==0:
				t1 = time.time()
				img_sec = 10 * BSIZE / (t1-t0)
				ls = tf.reduce_mean([_[0] for _ in lsacc])
				acc = tf.reduce_mean([_[1] for _ in lsacc])
				t0 = t1 
				print('Epoch:%d\tIter:%d\tLoss:%.6f\tAcc:%.6f\tSpeed:%.2f'%(ep, it, ls, acc, img_sec))

			if it%5000==0 and it>0:
				saver.save('./model/%d_%d.ckpt'%(ep,it))
				t0 = time.time()

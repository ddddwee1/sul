import tensorflow as tf 
import model3 as M 
import numpy as np 
import octresnet as resnet
import losspart
import datareader
import time 

class FaceResNet(M.Model):
	def initialize(self, num_classes):
		self.resnet = resnet.ResNet([64,64,128,256,512], [3, 4, 14, 3], 512, 0.75)
		self.classifier = losspart.MarginalCosineLayer(num_classes)

	def forward(self, x, label):
		# x = tf.image.resize(x, [128,128])
		feat = self.resnet(x)
		# feat = tf.nn.dropout(feat, 0.4)
		logits = self.classifier(feat, label, 1.0, 0.2, 0.0)
		logits = logits * 64
		return logits

BSIZE = 260 - 4
EPOCH = 30
data_reader = datareader.DataReader('img_list.txt', BSIZE)
tf.keras.backend.set_learning_phase(True)

def grad_loss(x, model):
	data, label = x
	with tf.GradientTape() as tape:
		out = model(data, label)
		wd = 0.0001
		w_reg = wd * sum([tf.reduce_sum(tf.square(w)) for w in model.trainable_variables]) 
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=label)) + w_reg
	acc = M.accuracy(out, label, one_hot=False) 
	grads = tape.gradient(loss, model.trainable_variables)
	return grads, [loss, acc]

# monitoring time
t0 = time.time()
# batch = data_reader.get_next()
LRV = 0.001
with tf.device('/cpu:0'):
	model = FaceResNet(data_reader.max_label + 1)
	LR = tf.Variable(LRV, trainable=False)
	optimizer = tf.optimizers.SGD(LR,0.9)
	saver = M.Saver(model, optimizer)
	saver.restore('./model/')
	

	_ = model(np.float32(np.ones([1,112,112,3])), np.float32(np.eye(data_reader.max_label+1)[0]))
	
	pt = M.ParallelTraining(model, optimizer, [0,1,2,3], grad_loss_fn=grad_loss) 

	for ep in range(EPOCH):
		if ep==1:
			LRV = 0.1
			LR.assign(LRV)
		if ep in [8,12,16]:
			LRV *= 0.1 
			LR.assign(LRV)
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
				print('Epoch:%d\tIter:%d\tLoss:%.6f\tAcc:%.6f\tSpeed:%.2f\tLR:%f'%(ep, it, ls, acc, img_sec, LRV))

			if it%5000==0 and it>0:
				saver.save('./model/%d_%d.ckpt'%(ep,it))
				t0 = time.time()

		saver.save('./model/%d_%d.ckpt'%(ep,it))

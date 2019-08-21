import hrnet 
import numpy as np 
import tensorflow as tf 
import model3 as M 
import data_reader
from tqdm import tqdm 

def grad_loss(x, model):
	data, label, mask = x
	# print(label.max())
	# print(label.min())
	with tf.GradientTape() as tape:
		out = model(data)
		# print(tf.reduce_max(out))
		# print(tf.reduce_mean(out))
		loss = tf.reduce_mean(tf.square(out - label))
		# loss = tf.reduce_mean(tf.square(out - label), axis=[2,3])
		# loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)

		# print(tf.reduce_max(out), tf.reduce_min(out))
	grads = tape.gradient(loss, model.trainable_variables)
	return grads, [loss]

class HRNET(M.Model):
	def initialize(self, num_pts):
		self.backbone = hrnet.ResNet()
		self.lastconv = M.ConvLayer(1, num_pts, usebias=False)
	def forward(self, x):
		feat = self.backbone(x)
		hmap = self.lastconv(feat)
		return hmap

tf.keras.backend.set_learning_phase(False)
net = HRNET(17)
M.Saver(net.backbone).restore('./hrnet/')

optim = tf.optimizers.Adam(0.0001, 0.5)
saver = M.Saver(net)
saver.restore('./model/')

# initialize
_ = np.zeros([1,256,256,3]).astype(np.float32)
net(_) 

# start training
reader = data_reader.data_reader(16)
MAX_ITER = 50000

bar = tqdm(range(MAX_ITER+1))
for i in bar:
	batch = reader.get_next()
	grads, lss = grad_loss(batch, net)
	optim.apply_gradients(M.zip_grad(grads, net.trainable_variables))
	bar.set_description('Loss:%.4f'%lss[0])
	if i%2000==0 and i>0:
		saver.save('./model/model.ckpt')

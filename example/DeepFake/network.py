import model3 as M 
import numpy as np 
import tensorflow as tf 
import data_reader

ENCODER_DIM = 1024
BSIZE = 64
MAX_ITER = 100000

class Encoder(M.Model):
	def initialize(self):
		self.c1 = M.ConvLayer(5, 128, stride=2, acitvation=M.PARAM_LRELU)
		self.c2 = M.ConvLayer(5, 256, stride=2, acitvation=M.PARAM_LRELU)
		self.c3 = M.ConvLayer(5, 512, stride=2, acitvation=M.PARAM_LRELU)
		self.c4 = M.ConvLayer(5,1024, stride=2, acitvation=M.PARAM_LRELU)
		self.fc1 = M.Dense(ENCODER_DIM)
	def forward(self,x):
		x = self.c1(x)
		x = self.c2(x)
		x = self.c3(x)
		x = self.c4(x)
		x = M.flatten(x)
		x = self.fc1(x)
		return x 

class Decoder(object):
	def initialize(self):
		self.fc1 = M.Dense(4*4*1024)
		self.up1 = M.ConvLayer(3, 512*4, acitvation=M.PARAM_LRELU)
		self.up2 = M.ConvLayer(3, 256*4, acitvation=M.PARAM_LRELU)
		self.up3 = M.ConvLayer(3, 128*4, acitvation=M.PARAM_LRELU)
		self.up4 = M.ConvLayer(3, 64*4, acitvation=M.PARAM_LRELU)
		self.c5 = M.ConvLayer(5, 3, acitvation=M.PARAM_SIGMOID)
	def forward(self, x):
		x = self.fc1(x)
		x = tf.reshape(x, [-1, 4,4,1024])
		x = self.up1(x)
		x = tf.nn.depth_to_space(x, 2)
		x = self.up2(x)
		x = tf.nn.depth_to_space(x, 2)
		x = self.up3(x)
		x = tf.nn.depth_to_space(x, 2)
		x = self.up4(x)
		x = tf.nn.depth_to_space(x, 2)
		x = self.c5(x)
		return x 
		
class FaceSwap(M.Model):
	def initialize(self):
		self.enc = Encoder()
		self.dec1 = Decoder()
		self.dec2 = Decoder()
	def forward(self,x):
		feat = self.enc(x)
		i1 = self.dec1(feat)
		i2 = self.dec2(feat)
		return i1,i2

def grad_loss(x, model):
	batch_a, label_a, batch_b, label_b = x
	with tf.GradientTape() as tape:
		pred_a, _ = model(batch_a)
		_, pred_b = model(batch_b)
		loss_a = tf.reduce_mean(tf.abs(label_a - pred_a))
		loss_b = tf.reduce_mean(tf.abs(label_b - pred_b))
		loss_total = loss_a + loss_b
	grad = tape.gradient(loss_total, model.trainable_variables)
	return grad, [loss_a, loss_b, loss_total]

swap_net = FaceSwap()
reader = data_reader.data_reader(BSIZE)
optim = tf.optimizers.Adam(0.0001)
saver = M.Saver(swap_net)
saver.restore('./model/')

for i in range(MAX_ITER+1):
	data_batch = reader.get_next()
	grad, losses = grad_loss(data_batch, swap_net)
	optim.apply_gradients(zip(grad, swap_net.trainable_variables))

	if i%10==0:
		print('Iter:%d\tLossA:%.4f\tLossB:%.4f\tLossTotal:%.4f'%(i, losses[0], losses[1], losses[2]))
		# TO-DO: Add visualization

	if i%2000==0 and i>0:
		saver.save('./model/model.ckpt')

import numpy as np 
import tensorflow as tf 
import model3 as M 
import datareader
from tqdm import tqdm 

sparse_values = np.float32(np.arange(2000, 8200, 100)) / 1000
sparse_values = sparse_values.reshape([1, 1, -1])
sparse_values = tf.convert_to_tensor(sparse_values)
# print(sparse_values)
# input()

class Network(M.Model):
	def initialize(self):
		self.c1 = M.ConvLayer1D(1, 512, pad='VALID', activation=M.PARAM_LRELU)
		self.c2 = M.ConvLayer1D(1, 512, pad='VALID', activation=M.PARAM_LRELU)
		self.c3 = M.ConvLayer1D(1, 512, pad='VALID', activation=M.PARAM_LRELU)
		self.c4 = M.ConvLayer1D(1, 512, pad='VALID', activation=M.PARAM_LRELU)
		self.c5 = M.ConvLayer1D(1, 512, pad='VALID', activation=M.PARAM_LRELU)
		self.c6 = M.ConvLayer1D(1, 512, pad='VALID', activation=M.PARAM_LRELU)
		self.c7 = M.ConvLayer1D(1, 62, usebias=False)

	def forward(self, x):
		x = self.c1(x)
		x = self.c2(x)
		x = self.c3(x)
		x = self.c4(x)
		x = self.c5(x)
		x = self.c6(x)
		x = self.c7(x)
		# print(x.shape)
		# input()
		return x

def soft_value(inp):
	# inp: bsize, temp, values
	# soft argmax
	inp = tf.nn.softmax(inp/ 100, axis=-1)
	# print(inp[0])
	result = inp * sparse_values
	result = tf.reduce_sum(result, axis=-1, keepdims=True)
	return result

def grad_loss(x, model):
	x, label = x
	with tf.GradientTape() as tape:
		out = model(x)
		# print(out[0])
		out = soft_value(out)
		# print(out[0])
		# print(label[0])
		# input()
		# loss = tf.reduce_mean(tf.abs(label - out))
		loss = tf.reduce_mean(tf.square(label - out))
	# loss2 = tf.reduce_mean(tf.square(label - out))
	grad = tape.gradient(loss, model.trainable_variables)
	return grad, [loss, out[0], label[0], loss]

def lr_decay(step):
	lr = 0.00002
	step = step/60000
	step = tf.math.floor(step)
	step = tf.math.pow(0.1, step)
	lr = lr * step 
	return lr 

BSIZE = 64
MAXITER = 240001

reader = datareader.data_reader(BSIZE, temp=0)
net = Network()

LR = M.LRScheduler(lr_decay)
# optim = tf.optimizers.SGD(LR, 0.8)

optim = tf.optimizers.Adam(LR, 0.5)
saver = M.Saver(net)
saver.restore('./model/')

meter = M.EMAMeter(0.95)
quadmeter = M.EMAMeter(0.95)

bar = tqdm(range(MAXITER))
for i in bar:
	bodycenter, scale, pts2d, z_value = reader.get_next()

	inp = reader.concat_all([bodycenter, scale, pts2d])
	label = z_value[:,0:1,0:1,0] / 1000
	batch = [inp, label]

	grad, loss = grad_loss(batch, net)
	optim.apply_gradients(zip(grad, net.trainable_variables))

	lr = lr_decay(optim.iterations)
	ls = meter.update(float(loss[0].numpy()))
	lsquad = quadmeter.update(float(loss[3].numpy()))
	bar.set_description('Loss:%.4f LR:%.1e OUT:%.4f LB:%.4f lossquad:%.4f'%(ls, lr, loss[1], loss[2], lsquad))

	if i%2000==0 and i>0:
		saver.save('./model/net%d.ckpt'%i)
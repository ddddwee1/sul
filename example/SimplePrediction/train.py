import model3 as M 
import numpy as np 
import tensorflow as tf 
import data_reader

class VariationalDrop(M.Model):
	# arxiv 1512.05287
	def initialize(self, drop_rate):
		self.drop_rate = drop_rate
	def _get_mask(self, shape):
		# (time, batch, dim)
		mask = np.random.choice(2, size=(1, shape[1], shape[2]), p=[1-self.drop_rate, self.drop_rate])
		return tf.convert_to_tensor(mask) 
	def forward(self, x):
		shape = x.shape()
		mask = self._get_mask(shape)
		x = x * mask
		return x 

class predNet(M.Model):
	def initialize(self):
		self.enc = M.Dense(128)
		self.LSTM = M.LSTM(128)
		self.dec = M.Dense(17*3)
	def forward(self, x):
		x = [self.enc(_) for _ in x]
		y = self.LSTM(x[:-1])
		y = [self.dec(_) for _ in y]
		return y 

def loss_grad(x, model):
	label = x[1:]
	with tf.GradientTape() as tape:
		out = model(x)
		# print(len(out))
		sub = [tf.square(o-l) for o,l in zip(out, label)]
		loss = tf.reduce_mean(sub)
	grad = tape.gradient(loss, model.trainable_variables)
	return grad, [loss]


reader = data_reader.data_reader()

model = predNet()
optim = tf.optimizers.Adam(0.001)

saver = M.Saver(model)
ITER = 10000
for i in range(ITER + 1):
	data = reader.get_next(32, 16)
	grad, ls = loss_grad(data, model)
	optim.apply_gradients(zip(grad, model.trainable_variables))
	if i%10==0:
		print('Iter:%d\tLoss:%.4f'%(i, ls[0]))
	if i%2000==0:
		saver.save('./model/model.ckpt')

import tensorflow as tf 
import layers2 as L 
import modeleag as M 
import numpy as np 
tf.enable_eager_execution()

class model(M.Model):
	def initialize(self):
		self.conv1 = L.conv2D(None, 3, 16)
		self.conv2 = L.conv2D(None, 3, 32)

	def forward(self,x):
		x = self.conv1(x)
		x = self.conv2(x)
		return x

m = model()

def get_ls(x):
	with tf.GradientTape() as tape:
		r = m(x)
		ls = tf.reduce_mean(tf.abs(r - tf.ones_like(r)))
	return ls, tape

optim = tf.train.AdamOptimizer(0.01)
saver = M.Saver(m, optim)
saver.restore('./mod/')
print('loaded')

for i in range(10):
	x = np.random.uniform(size=[1,10,10,3])
	x = np.float32(x)
	ls, tape = get_ls(x)
	grad = tape.gradient(ls, m.variables)

	optim.apply_gradients(zip(grad, m.variables), global_step=tf.train.get_or_create_global_step())
	print(ls.numpy())

	input()

saver.save('./mod/tst')

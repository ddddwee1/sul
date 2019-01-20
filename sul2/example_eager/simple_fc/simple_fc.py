import tensorflow as tf 
import layers2 as L 
import modeleag as M 
import numpy as np 
tf.enable_eager_execution()

class generator_3d(M.Model):
	def initialize(self):
		self.activ = L.activation(M.PARAM_LRELU)
		self.l1 = L.fcLayer(1024)
		self.l2 = L.fcLayer(1024)
		self.l3 = L.fcLayer(1024)
		self.l4 = L.fcLayer(17)

	def forward(self, x):
		l1 = self.activ(self.l1(x))
		l2 = self.activ(self.l2(l1))
		l3 = self.activ(self.l3(l2) + l1)
		l4 = self.l4(l3)
		return l4 

mod = generator_3d()

saver = M.Saver(mod)
saver.restore('./model/')

x = np.ones([1,34], dtype=np.float32)
y = mod(x)

print(y)

# saver = M.Saver(mod)
# saver.save('./model/generator_3d.ckpt')

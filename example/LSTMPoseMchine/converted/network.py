import model3 as M 
import numpy as np 
import tensorflow as tf 

def get_conv(name):
	return None

def get_conv2(name):
	return None 

class Stage0(M.Model):
	def initialize(self):
		# init encoding 
		self.c1_s1 = M.ConvLayer(9, 128, pad='SAME_LEFT', activation=M.PARAM_RELU)
		self.p1_s1 = M.MaxPool(3, 2, pad='VALID')
		self.c2_s1 = M.ConvLayer(9, 128, pad='SAME_LEFT', activation=M.PARAM_RELU)
		self.p2_s1 = M.MaxPool(3, 2, pad='VALID')
		self.c3_s1 = M.ConvLayer(9, 128, pad='SAME_LEFT', activation=M.PARAM_RELU)
		self.p3_s1 = M.MaxPool(3, 2, pad='VALID')
		self.c4_s1 = M.ConvLayer(5, 32, pad='SAME_LEFT', activation=M.PARAM_RELU)
		self.c5_s1 = M.ConvLayer(9, 512, pad='SAME_LEFT', activation=M.PARAM_RELU)
		self.c6_s1 = M.ConvLayer(1, 512, activation=M.PARAM_RELU)
		self.c7_s1 = M.ConvLayer(1, 15)

		# frame encoding 
		self.c1_s2 = M.ConvLayer(9, 128, pad='SAME_LEFT', activation=M.PARAM_RELU)
		self.p1_s2 = M.MaxPool(3, 2, pad='VALID')
		self.c2_s2 = M.ConvLayer(9, 128, pad='SAME_LEFT', activation=M.PARAM_RELU)
		self.p2_s2 = M.MaxPool(3, 2, pad='VALID')
		self.c3_s2 = M.ConvLayer(9, 128, pad='SAME_LEFT', activation=M.PARAM_RELU)
		self.p3_s2 = M.MaxPool(3, 2, pad='VALID')
		self.c4_s2 = M.ConvLayer(5, 32, pad='SAME_LEFT', activation=M.PARAM_RELU)

		# center map
		self.pool = M.AvgPool(9,8, pad='VALID')

		# LSTM0
		self.g = M.ConvLayer(3, 48, pad='SAME_LEFT')
		self.gb = tf.Variable(np.zeros([48]).astype(np.float32),validate_shape=False)
		self.i = M.ConvLayer(3, 48, pad='SAME_LEFT')
		self.ib = tf.Variable(np.zeros([48]).astype(np.float32),validate_shape=False)
		self.o = M.ConvLayer(3, 48, pad='SAME_LEFT')
		self.ob = tf.Variable(np.zeros([48]).astype(np.float32),validate_shape=False)

		# decoder branch 
		self.mc1 = M.ConvLayer(11, 128, pad='SAME_LEFT', activation=M.PARAM_RELU)
		self.mc2 = M.ConvLayer(11, 128, pad='SAME_LEFT', activation=M.PARAM_RELU)
		self.mc3 = M.ConvLayer(11, 128, pad='SAME_LEFT', activation=M.PARAM_RELU)
		self.mc4 = M.ConvLayer(1, 128, activation=M.PARAM_RELU)
		self.mc5 = M.ConvLayer(1, 15)


	def forward(self, dt1, dt2, centermap):
		#init enc
		e = dt1 
		e = self.c1_s1(e)
		e = tf.pad(e, [[0,0],[0,1],[0,1],[0,0]], mode='SYMMETRIC')
		e = self.p1_s1(e)
		e = self.c2_s1(e)
		e = tf.pad(e, [[0,0],[0,1],[0,1],[0,0]], mode='SYMMETRIC')
		e = self.p2_s1(e)
		e = self.c3_s1(e)
		e = tf.pad(e, [[0,0],[0,1],[0,1],[0,0]], mode='SYMMETRIC')
		e = self.p3_s1(e)
		e = self.c4_s1(e)
		e = self.c5_s1(e)
		e = self.c6_s1(e)
		e = self.c7_s1(e)

		# frame encoding 
		f = dt2 
		f = self.c1_s2(f)
		f = tf.pad(f, [[0,0],[0,1],[0,1],[0,0]], mode='SYMMETRIC')
		f = self.p1_s2(f)
		f = self.c2_s2(f)
		f = tf.pad(f, [[0,0],[0,1],[0,1],[0,0]], mode='SYMMETRIC')
		f = self.p2_s2(f)
		f = self.c3_s2(f)
		f = tf.pad(f, [[0,0],[0,1],[0,1],[0,0]], mode='SYMMETRIC')
		f = self.p3_s2(f)
		f = self.c4_s2(f)

		# centermap pooling 
		x = tf.pad(centermap, [[0,0],[0,1],[0,1],[0,0]], mode='SYMMETRIC')
		x = self.pool(x)

		# LSTM branch 
		x = tf.concat([f, e, x], axis=-1)
		g = self.g(x) + self.gb 
		i = self.i(x) + self.ib
		o = self.o(x) + self.ob

		g = tf.tanh(g)
		i = tf.sigmoid(i)
		o = tf.sigmoid(o)

		c = g * i 
		h = o * tf.tanh(c)

		# decoder branch 
		x = self.mc1(h)
		x = self.mc2(x)
		x = self.mc3(x)
		x = self.mc4(x)
		out = self.mc5(x)

		return out 

class Stage1(M.Model):
	def initialize(self):
		# frame encoding 
		self.c1_s2 = M.ConvLayer(9, 128, pad='SAME_LEFT', activation=M.PARAM_RELU)
		self.p1_s2 = M.MaxPool(3, 2, pad='VALID')
		self.c2_s2 = M.ConvLayer(9, 128, pad='SAME_LEFT', activation=M.PARAM_RELU)
		self.p2_s2 = M.MaxPool(3, 2, pad='VALID')
		self.c3_s2 = M.ConvLayer(9, 128, pad='SAME_LEFT', activation=M.PARAM_RELU)
		self.p3_s2 = M.MaxPool(3, 2, pad='VALID')
		self.c4_s2 = M.ConvLayer(5, 32, pad='SAME_LEFT', activation=M.PARAM_RELU)

		# center map
		self.pool = M.AvgPool(9,8, pad='VALID')

		# lstm
		self.gx = M.ConvLayer(3, 48, pad='SAME_LEFT')
		self.gh = M.ConvLayer(3, 48, pad='SAME_LEFT')
		self.gb = tf.Variable(np.zeros([48]).astype(np.float32),validate_shape=False)
		self.fx = M.ConvLayer(3, 48, pad='SAME_LEFT')
		self.fh = M.ConvLayer(3, 48, pad='SAME_LEFT')
		self.fb = tf.Variable(np.zeros([48]).astype(np.float32),validate_shape=False)
		self.ox = M.ConvLayer(3, 48, pad='SAME_LEFT')
		self.oh = M.ConvLayer(3, 48, pad='SAME_LEFT')
		self.ob = tf.Variable(np.zeros([48]).astype(np.float32),validate_shape=False)
		self.ix = M.ConvLayer(3, 48, pad='SAME_LEFT')
		self.ih = M.ConvLayer(3, 48, pad='SAME_LEFT')
		self.ib = tf.Variable(np.zeros([48]).astype(np.float32),validate_shape=False)

		# decoder branch 
		self.mc1 = M.ConvLayer(11, 128, pad='SAME_LEFT', activation=M.PARAM_RELU)
		self.mc2 = M.ConvLayer(11, 128, pad='SAME_LEFT', activation=M.PARAM_RELU)
		self.mc3 = M.ConvLayer(11, 128, pad='SAME_LEFT', activation=M.PARAM_RELU)
		self.mc4 = M.ConvLayer(1, 128, activation=M.PARAM_RELU)
		self.mc5 = M.ConvLayer(1, 15)

	def forward(self, x, hmap, centermap, h, c):
		# frame encoding 
		f = x 
		f = self.c1_s2(f)
		f = tf.pad(f, [[0,0],[0,1],[0,1],[0,0]], mode='SYMMETRIC')
		f = self.p1_s2(f)
		f = self.c2_s2(f)
		f = tf.pad(f, [[0,0],[0,1],[0,1],[0,0]], mode='SYMMETRIC')
		f = self.p2_s2(f)
		f = self.c3_s2(f)
		f = tf.pad(f, [[0,0],[0,1],[0,1],[0,0]], mode='SYMMETRIC')
		f = self.p3_s2(f)
		f = self.c4_s2(f)

		# centermap pooling 
		ce = tf.pad(centermap, [[0,0],[0,1],[0,1],[0,0]], mode='SYMMETRIC')
		ce = self.pool(ce)

		# lstm branch 
		x = tf.concat([f, hmap, ce], axis=-1)
		gx = self.gx(x)
		gh = self.gh(h)

		ox = self.ox(x)
		oh = self.oh(h)

		fx = self.fx(x)
		fh = self.fh(h)

		ix = self.ix(x)
		ih = self.ih(h)

		g = tf.tanh(gx + gh + self.gb)
		o = tf.sigmoid(ox + oh + self.ob)
		i = tf.sigmoid(ix + ih + self.ib)
		f = tf.sigmoid(fx + fh + self.fb)

		c = f*c + i*g 
		h = o * tf.tanh(c)

		# decoder branch 
		x = self.mc1(h)
		x = self.mc2(x)
		x = self.mc3(x)
		x = self.mc4(x)
		out = self.mc5(x)
		return out 

class ModelBundle(M.Model):
	def initialize(self):
		self.s0 = Stage0()
		self.s1 = Stage1()

if __name__=='__main__':
	mods = ModelBundle()
	saver = M.Saver(mods)
	saver.restore('./LSTMPM/')

	mod = mods.s0
	x = np.ones([1,368,368,3]).astype(np.float32)
	cent = np.ones([1,368,368,1]).astype(np.float32)
	x = mod(x, x, cent)
	out = np.transpose(x,[0,3,1,2])
	print(out)
	print(out.shape)
	input('Test deploy1 finished. Input for testing deploy2')

	mod = mods.s1
	x = np.ones([1,368,368,3]).astype(np.float32)
	cent = np.ones([1,368,368,1]).astype(np.float32)
	h = c = np.ones([1,46,46, 48]).astype(np.float32)
	hmap = np.ones([1,46,46, 15]).astype(np.float32)

	x[:,-1] = 0

	x = mod(x, hmap, cent, h, c)
	out = np.transpose(x,[0,3,1,2])
	print(out)
	print(out.shape)
	input('Test deploy2 finished. Input for saving converted weights ')

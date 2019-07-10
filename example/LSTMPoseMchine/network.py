import model3 as M 
import tensorflow as tf 

class Net1(M.Model):
	def initialize(self, out):
		self.c1 = M.ConvLayer(9, 128, activation=M.PARAM_RELU)
		self.p1 = M.MaxPool(3, 2)
		self.c2 = M.ConvLayer(9, 128, activation=M.PARAM_RELU)
		self.p2 = M.MaxPool(3, 2)
		self.c3 = M.ConvLayer(9, 128, activation=M.PARAM_RELU)
		self.p3 = M.MaxPool(3, 2)
		self.c4 = M.ConvLayer(5, 32, activation=M.PARAM_RELU)
		self.c5 = M.ConvLayer(9, 512 ,activation=M.PARAM_RELU)
		self.c6 = M.ConvLayer(1, 512, activation=M.PARAM_RELU)
		self.c7 = M.ConvLayer(1, out)
	def forward(self, x):
		x = self.c1(x)
		x = self.p1(x)
		x = self.c2(x)
		x = self.p2(x)
		x = self.c3(x)
		x = self.p3(x)
		x = self.c4(x)
		x = self.c5(x)
		x = self.c7(self.c6(x))
		return x 

class Net2(M.Model):
	def initialize(self):
		self.c1 = M.ConvLayer(9, 128, activation=M.PARAM_RELU)
		self.p1 = M.MaxPool(3, 2)
		self.c2 = M.ConvLayer(9, 128, activation=M.PARAM_RELU)
		self.p2 = M.MaxPool(3, 2)
		self.c3 = M.ConvLayer(9, 128, activation=M.PARAM_RELU)
		self.p3 = M.MaxPool(3, 2)
		self.c4 = M.ConvLayer(5, 32, activation=M.PARAM_RELU)
	def forward(self, x):
		x = self.c1(x)
		x = self.p1(x)
		x = self.c2(x)
		x = self.p2(x)
		x = self.c3(x)
		x = self.p3(x)
		x = self.c4(x)
		return x 

class Net3(M.Model):
	def initialize(self, out):
		self.c1 = M.ConvLayer(11, 128, activation=M.PARAM_RELU)
		self.c2 = M.ConvLayer(11, 128 ,activation=M.PARAM_RELU)
		self.c3 = M.ConvLayer(11, 128, activation=M.PARAM_RELU)
		self.c4 = M.ConvLayer(1, 128, activation=M.PARAM_RELU)
		self.c5 = M.ConvLayer(1, out)
	def forward(self, x):
		x = self.c1(x)
		x = self.c2(x)
		x = self.c3(x)
		x = self.c4(x)
		x = self.c5(x)
		return x 

class LSTM0(M.Model):
	def initialize(self, chn):
		self.g = M.ConvLayer(3, chn)
		self.i = M.ConvLayer(3, chn)
		self.o = M.ConvLayer(3, chn)

	def forward(self, x):
		g = self.g(x)
		i = self.i(x)
		o = self.o(x)

		g = tf.tanh(g)
		i = tf.sigmoid(i)
		o = tf.sigmoid(o)

		cell = g * i 
		h = o * tf.tanh(cell)
		return c, h

class ConvLSTM(M.Model):
	def initialize(self, chn):
		self.gx = M.ConvLayer(3, chn)
		self.gh = M.ConvLayer(3, chn)
		self.fx = M.ConvLayer(3, chn)
		self.fh = M.ConvLayer(3, chn)
		self.ox = M.ConvLayer(3, chn)
		self.oh = M.ConvLayer(3, chn)
		self.gx = M.ConvLayer(3, chn)
		self.gh = M.ConvLayer(3, chn)

	def forward(self, x, c, h):
		gx = self.gx(x)
		gh = self.gh(h)

		ox = self.ox(x)
		oh = self.oh(h)

		fx = self.fx(x)
		fh = self.fh(h)

		gx = self.gx(x)
		gh = self.gh(h)

		g = tf.tanh(gx + gh)
		o = tf.sigmoid(ox + oh)
		i = tf.sigmoid(ix + ih)
		f = tf.sigmoid(fx + fh)

		cell = f*c + i*g 
		h = o * tf.tanh(cell)
		return cell, h 

class LSTMPM(M.Model):
	def initialize(self):
		points = 17
		lstmchannel = 48
		self.net1 = Net1(points)
		self.net2 = Net2()
		self.net3 = Net3(points)
		self.lstm0 = LSTM0(lstmchannel)
		self.lstm = ConvLSTM(lstmchannel)
		self.pool = M.AvgPool(9, 8)

	def stage1(self, x, cmap):
		init_map = self.net1(x)
		feat = self.net2(x)
		pool_center = self.pool(cmap)

		x = tf.concat([init_map, feat, pool_center], axis=-1)
		c, h = self.LSTM0(x)
		hmap = self.net3(h)
		return init_map, hmap, c, h 

	def stage2(self, x, cmap, hmap, c, h):
		feat = self.net2(x)
		pool_center = self.pool(cmap)
		x = tf.concat([hmap, feat, pool_center], axis=-1)
		c, h = self.lstm(x, c, h)
		hmap = self.net3(hmap)
		return hmap, c, h 

	def forward(self, x, cmap):
		init_map, hmap, c, h = self.stage1(x[0], cmap[0])

		res = [init_map, hmap]
		for i in range(x.shape[0]-1):
			hmap, c, h = stage2(x[i+1], cmap[i+1], hmap, c, h)
			res.append(hmap)
		return res 

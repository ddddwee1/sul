import model3 as M 
import tensorflow as tf 
import config 

class Drop(M.Model):
	def initialize(self, drop_rate):
		self.keep_rate = 1 - drop_rate
	def forward(self, x):
		random_tensor = tf.random_uniform([tf.shape(x)[0],1,1,1]) + self.keep_rate
		binary = tf.floor(random_tensor)
		x = x * binary
		return x 

class MBConvBlock(M.Model):
	def initialize(self, input_filters, output_filters, expand_ratio, kernel_size, stride, se_ratio = 0.25, drop_rate=0.2):
		self.stride = stride
		self.input_filters = input_filters
		self.output_filters = output_filters
		outchn = input_filters * expand_ratio 
		if expand_ratio!=1:
			self.expand_conv = M.ConvLayer(1, outchn, activation=M.PARAM_SWISH, usebias=False, batch_norm=True)
		self.dwconv = M.DWConvLayer(kernel_size, 1, activation=M.PARAM_SWISH, stride=stride, usebias=False, batch_norm=True)

		# se 
		se_outchn = max(1, int(input_filters * se_ratio))
		self.se_gap = M.GlobalAvgPool()
		self.se_reduce = M.ConvLayer(1, se_outchn, activation=M.PARAM_SWISH)
		self.se_expand = M.ConvLayer(1, outchn, activation=M.PARAM_SIGMOID)

		# output 
		outchn = output_filters
		self.proj_conv = M.ConvLayer(1, outchn, usebias=False, batch_norm=True)

		if drop_rate:
			self.drop = Drop(drop_rate)

	def forward(self, x):
		origin = x
		if hasattr(self, 'expand_conv'):
			x = self.expand_conv(x)
		x = self.dwconv(x)
		# se
		inp = x
		x = self.se_gap(x)
		x = self.se_expand(self.se_reduce(x))
		x = x * inp 
		# out 
		x = self.proj_conv(x)
		if self.stride==1 and self.input_filters==self.output_filters:
			if hasattr(self, 'drop') and tf.keras.backend.learning_phase():
				x = self.drop(x)
			x = tf.add(x, origin)
		return x 

class EffNet(M.Model):
	def initialize(self):
		self.c1 = M.ConvLayer(config.ksize[0], config.out_chnnels[0], stride=config.strides[0], activation=M.PARAM_SWISH, usebias=False, batch_norm=True)
		self.blocks = []
		for r,k,i,o,s,e in zip(config.repeats, config.ksize[1:], config.out_chnnels[:-1], config.out_chnnels[1:], config.strides[1:], config.expansions):
			for num_blk in range(r):
				self.blocks.append(MBConvBlock(i, o, e, 1 if num_blk==0 else k, s))

		self.emb_bn = M.BatchNorm()
		self.fc1 = M.Dense(config.emb_size, batch_norm=True)
	def forward(self, x):
		x = self.c1(x)
		for block in self.blocks:
			x = block(x)

		x = M.flatten(x)
		x = self.emb_bn(x)
		if tf.keras.backend.learning_phase():
			x = tf.nn.dropout(x, 0.4)
		x = self.fc1(x)
		return x

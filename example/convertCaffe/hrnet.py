import tensorflow as tf 
import model3 as M 
import layers3 as L 
import numpy as np 

EXPAND = 6

class MBConvBlock(M.Model):
	def initialize(self, input_filters, output_filters, expand_ratio, kernel_size, stride, se_ratio = 0.25):
		self.stride = stride
		self.input_filters = input_filters
		self.output_filters = output_filters
		outchn = input_filters * expand_ratio 
		if expand_ratio!=1:
			self.expand_conv = M.ConvLayer(1, outchn, activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
		self.dwconv = M.DWConvLayer(kernel_size, 1, activation=M.PARAM_LRELU, stride=stride, usebias=False, batch_norm=True)

		# se 
		se_outchn = max(1, int(input_filters * se_ratio))
		self.se_gap = M.GlobalAvgPool()
		self.se_reduce = M.ConvLayer(1, se_outchn, activation=M.PARAM_LRELU)
		self.se_expand = M.ConvLayer(1, outchn, activation=M.PARAM_SIGMOID)

		# output 
		outchn = output_filters
		self.proj_conv = M.ConvLayer(1, outchn, usebias=False, batch_norm=True)

	def forward(self, x):
		origin = x
		if hasattr(self, 'expand_conv'):
			x = self.expand_conv(x)
		x = self.dwconv(x)
		# se
		inp = x
		x = self.se_gap(x)
		x = self.se_expand(self.se_reduce(x))
		x = L.BroadcastMUL()([x, inp])
		# out 
		x = self.proj_conv(x)
		if self.stride==1 and self.input_filters==self.output_filters:
			x = L.SUM()([x, origin])
		x = L.activation(M.PARAM_RELU)(x)
		return x 


class ResBlock(M.Model):
	def initialize(self, inp, out, stride, num_units):
		self.units = []
		for i in range(num_units):
			self.units.append(MBConvBlock(inp if i==0 else out, out, stride=stride if i==0 else 1, expand_ratio=1, kernel_size=3))
	def forward(self, x):
		for unit in self.units:
			x = unit(x)
		return x 

class ResBasicBlock(M.Model):
	def initialize(self, out, num_units):
		self.units = []
		for i in range(num_units):
			self.units.append(MBConvBlock(out, out, stride=1, expand_ratio=EXPAND, kernel_size=5))
	def forward(self, x):
		for unit in self.units:
			x = unit(x)
		return x 

class Transition(M.Model):
	def initialize(self, outchns, strides):
		self.trans = []
		for i,(o,s) in enumerate(zip(outchns,strides)):
			if o is None or s is None:
				self.trans.append(None)
			elif s==1:
				self.trans.append(M.ConvLayer(3,o, stride=s, activation=M.PARAM_LRELU, usebias=False, batch_norm=True))
			else:
				self.trans.append(M.ConvLayer(3,o, stride=s, activation=M.PARAM_LRELU, usebias=False, batch_norm=True))

	def forward(self, x):
		results = []
		for i,t in enumerate(self.trans):
			if t is None:
				results.append(x[i])
			else:
				results.append(t(x[-1]))
		return results

class FuseDown(M.Model):
	def initialize(self, steps, inp, o):
		self.mods = []
		for i in range(steps):
			if i==(steps-1):
				self.mods.append(M.ConvLayer(3, o, stride=2, batch_norm=True, usebias=False))
			else:
				self.mods.append(M.ConvLayer(3, inp, stride=2, activation=M.PARAM_LRELU, batch_norm=True, usebias=False))
	def forward(self, x):
		for m in self.mods:
			x = m(x)
		return x 

class FuseUp(M.Model):
	def initialize(self, o):
		self.c1 = M.ConvLayer(1, o, batch_norm=True, usebias=False)
	def forward(self, x, target_shape):
		x = self.c1(x)
		print(target_shape)
		scale = x[0].get_shape().as_list()
		try:
			scale = int(target_shape[1].numpy()) // int(scale[1])
		except:
			scale = int(target_shape[1]) // int(scale[1])
		# x = tf.image.resize(x, target_shape, method='nearest')
		x = L.NNUpSample2D(scale)(x)
		return x 

class Fuse(M.Model):
	def initialize(self,outchns):
		branches = []
		for i in range(len(outchns)): # target
			branch = []
			for j in range(len(outchns)): # source
				if i==j:
					branch.append(None)
				elif i<j:
					branch.append(FuseUp(outchns[i]))
				else:
					branch.append(FuseDown(i-j, outchns[j], outchns[i]))
			branches.append(branch)
		self.branches = branches
	def forward(self, x):
		out = []
		for i in range(len(self.branches)): # target
			branch_out = []
			for j in range(len(self.branches)): # source
				if i==j:
					branch_out.append(x[i])
				elif i<j:
					branch_out.append(self.branches[i][j](x[j] , target_shape=tf.shape(x[i][0])[1:3]))
				else:
					branch_out.append(self.branches[i][j](x[j]))
			branch_out = L.SUM()(branch_out)
			out.append(L.activation(M.PARAM_RELU)(branch_out))
		return out 

class FuseLast(M.Model):
	def initialize(self, outchns):
		self.c1 = FuseUp(outchns[0])
		self.c2 = FuseUp(outchns[0])
		self.c3 = FuseUp(outchns[0])
	def forward(self, x):
		out = [x[0]]
		out.append(self.c1(x[1], tf.shape(x[0])[1:3]))
		out.append(self.c2(x[2], tf.shape(x[0])[1:3]))
		out.append(self.c3(x[3], tf.shape(x[0])[1:3]))
		out = L.SUM()(out)
		out = L.activation(M.PARAM_RELU)(out)
		return out 

class FuseLastV2(M.Model):
	def initialize(self, outchns):
		self.c1 = FuseDown(3, outchns[0], outchns[3])
		self.c2 = FuseDown(2, outchns[1], outchns[3])
		self.c3 = FuseDown(1, outchns[2], outchns[3])
	def forward(self, x):
		out = [x[3]]
		out.append(self.c1(x[0]))
		out.append(self.c2(x[1]))
		out.append(self.c3(x[2]))
		out = L.CONCAT()(out)
		return out 

class Stage(M.Model):
	def initialize(self, outchns, strides, num_units, num_fuses, d=False):
		self.d = d 
		self.num_fuses = num_fuses
		self.transition = Transition(outchns, strides)
		self.blocks = []
		self.fuses = []
		for j in range(num_fuses):
			block = []
			for i in range(len(outchns)):
				block.append(ResBasicBlock(outchns[i], num_units))
			self.blocks.append(block)
			if not (self.d and j==(self.num_fuses-1)):
				self.fuses.append(Fuse(outchns))

	def forward(self, x ):
		x = self.transition(x)
		for i in range(self.num_fuses):
			out = []
			for o,b in zip(x, self.blocks[i]):
				out.append(b(o))
			if not (self.d and i==(self.num_fuses-1)):
				x = self.fuses[i](out)
			else:
				x = out 
		return x 

class ResNet(M.Model):
	def initialize(self, emb_dim):
		# self.c1 = M.ConvLayer(5, 32, stride=2, batch_norm=True ,usebias=False)
		self.c1 = M.ConvLayer(5, 32, stride=2, activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
		self.layer1 = ResBlock(32, 32, 2, 4)
		self.stage1 = Stage([16, 32], [1, 2], 4, 1)
		self.stage2 = Stage([16, 32, 64], [None, None, 2], 4, 4)
		self.stage3 = Stage([16, 32, 64, 128], [None,None,None,2], 4, 3, d=True)
		# # self.lastfuse = FuseLast([16,32,64,128])
		self.lastfuse = FuseLastV2([16,32,64,128])
		self.emb_bn = M.BatchNorm()
		self.emb = M.Dense(emb_dim, usebias=False, batch_norm=True, map_shape=[4,4,512])

	def forward(self, x):
		x = self.c1(x)
		x = self.layer1(x)
		x = self.stage1([x,x])
		x = self.stage2(x)
		x = self.stage3(x)
		x = self.lastfuse(x)
		x = self.emb_bn(x)
		x = self.emb(M.flatten(x))
		return x[0]

# tf.keras.backend.set_learning_phase(False)

# net = ResNet(512)
# saver = M.Saver(net)
# saver.restore('./model/')

# x = np.ones([1,256,256,3]).astype(np.float32)
# y = net(x)
# print(tf.transpose(y, [0,3,1,2]))


import model3 as M 
import numpy as np 
import tensorflow as tf 

#### OP Blocks ####
class Zero(M.Model):
	def initialize(self, stride):
		self.stride= stride
	def forward(self, x):
		if self.stride==1:
			return x * 0.
		else:
			x = x[:,::self.stride, ::self.stride, :] * 0
			return x 

class MP33(M.Model):
	def initialize(self, stride, bn=False):
		self.p = M.MaxPool(3, stride)
		self.isbn = bn 
		if bn:
			self.bn = M.BatchNorm()
	def forward(self, x):
		x = self.p(x)
		if self.isbn:
			x = self.bn(x)
		return x

class AP33(M.Model):
	def initialize(self, stride, bn=False):
		self.p = M.AvgPool(3, stride)
		self.isbn = bn 
		if bn:
			self.bn = M.BatchNorm()
	def forward(self, x):
		x = self.p(x)
		if self.isbn:
			x = self.bn(x)
		return x

class Skip(M.Model):
	def initialize(self, out, stride):
		self.stride = stride
		if stride!=1:
			self.rd = FactReduce(out)
	def forward(self, x):
		if self.stride==1:
			return x 
		else:
			return self.rd(x)

class SPConv33(M.Model):
	def initialize(self, out, stride):
		self.c1 = M.DWConvLayer(3, 1, usebias=False)
		self.c2 = M.ConvLayer(1, out, usebias=False, batch_norm=True)
	def forward(self, x):
		x = tf.nn.relu(x)
		x = self.c1(x)
		x = self.c2(x)
		return x 

class SPConv55(M.Model):
	def initialize(self, out, stride):
		self.c1 = M.DWConvLayer(5, 1, usebias=False)
		self.c2 = M.ConvLayer(1, out, usebias=False, batch_norm=True)
	def forward(self, x):
		x = tf.nn.relu(x)
		x = self.c1(x)
		x = self.c2(x)
		return x 

class DilConv33(M.Model):
	def initialize(self, out, stride):
		self.c1 = M.ConvLayer(3, out, stride=stride, dilation_rate=2, batch_norm=True, usebias=False)
	def forward(self, x):
		x = tf.nn.relu(x)
		x = self.c1(x)
		return x 

class DilConv55(M.Model):
	def initialize(self, out, stride):
		self.c1 = M.ConvLayer(5, out, stride=stride, dilation_rate=2, batch_norm=True, usebias=False)
	def forward(self, x):
		x = tf.nn.relu(x)
		x = self.c1(x)
		return x 

class ASPP(M.Model):
	def initialize(self, out, dilation):
		self.c11 = M.ConvLayer(1, out, usebias=False, batch_norm=True)
		self.c33 = M.ConvLayer(3, out, usebias=False, batch_norm=True, dilation_rate=dilation)
		self.convp = M.ConvLayer(1, out, usebias=False, batch_norm=True)
		self.concat_conv = M.ConvLayer(1, out, usebias=False, batch_norm=True)
		self.GP = M.GlobalAvgPool()
	def forward(self, x):
		x11 = self.c11(x)
		x33 = self.c33(x)
		pool = self.GP(x)
		up = tf.ones_like(x) * pool 
		concat = tf.concat([x11, x33, up], axis=-1)
		out = self.concat_conv(concat)
		return out 

#### functional blocks ####

class FuseDown(M.Model):
	def initialize(self, steps, inp, o):
		self.mods = []
		for i in range(steps):
			if i==(steps-1):
				self.mods.append(M.ConvLayer(3, o, stride=2, pad='SAME_LEFT', batch_norm=True, usebias=False))
			else:
				self.mods.append(M.ConvLayer(3, inp, stride=2, pad='SAME_LEFT', activation=M.PARAM_RELU, batch_norm=True, usebias=False))
	def forward(self, x):
		for m in self.mods:
			x = m(x)
		return x 

class FactReduce(M.Model):
	def initialize(self, out):
		self.c1 = M.ConvLayer(1, out//2, stride=2, usebias=False)
		self.c2 = M.ConvLayer(1, out//2, stride=2, usebias=False)
		self.bn = M.BatchNorm()
	def forward(self, x):
		x = tf.nn.relu(x)
		x1 = self.c1(x)
		x2 = self.c2(x[:,1:,1:,:])
		x = tf.concat([x1, x2], axis=-1)
		x = self.bn(x)
		return x 

class FactIncrease(M.Model):
	def initialize(self, out):
		self.up = M.BilinearUpSample(2)
		self.c1 = M.ConvLayer(1, out, usebias=False, batch_norm=True)
	def forward(self, x):
		x = self.up(x)
		x = tf.nn.relu(x)
		x = self.c1(x)
		return x 

class MixedOp(M.Model):
	def initialize(self, out, stride):
		ops = []
		ops.append(Zero(stride))
		ops.append(MP33(stride, True))
		ops.append(AP33(stride, True))
		ops.append(Skip(out, stride))
		ops.append(SPConv33(out, stride))
		ops.append(SPConv55(out, stride))
		ops.append(DilConv33(out, stride))
		ops.append(DilConv55(out, stride))
		self.ops = ops 
	def forward(self, x, weights):
		return sum(w*op(x) for w,op in zip(weights,self.ops))

class CellBuilder(M.Model):
	def initialize(self, step, multiplier, c_pp, c_p, c, rate):
		'''
		cpp: prev_prev channel num
		cp: prev channel num 
		c: current channel num 
		'''
		self.multiplier = multiplier
		self.step = step 
		if c_pp != -1:
			self.preprocess0 = M.ConvLayer(1, c, batch_norm=True, usebias=False)
		
		if rate ==2:
			self.preprocess1 = FactReduce(c)
		elif rate==0:
			self.preprocess1 = FactIncrease(c)
		else:
			self.preprocess1 = M.ConvLayer(c, 1, activation=M.PARAM_RELU, batch_norm=True, usebias=False)

		_ops = []
		if c_pp != -1:
			for i in range(step):
				for j in range(2+i):
					# stride = 1 
					_ops.append(MixedOp(c, 1))
		else:
			for i in range(step):
				for j in range(1+i):
					_ops.append(MixedOp(c, 1))
		self._ops = _ops
		self.conv_last = M.ConvLayer(1, c, activation=M.PARAM_RELU, batch_norm=True, usebias=False)

	def forward(self, s0, s1, w):
		# The following is not compatible with tf.graph (tf.function)
		# TO-DO: will change to graph compatible format for multi-gpu processing
		if s0 is not None:
			s0 = self.preprocess0(s0)
		s1 = self.preprocess1(s1)
		if s0 is not None:
			states = [s0, s1]
		else:
			states = [s1]
		offset = 0
		for i in range(self.step):
			s = sum(self._ops[offset+j](h, w[offset+j]) for j,h in enumerate(states))
			offset += len(states)
			states.append(s)
		concat_feat = tf.concat(states[-self.multiplier:], axis=-1)
		out = self.conv_last(concat_feat)
		return out 

#### Network Strcuture ####

# class Stems(M.Model):
# 	def initialize(self):
		# self.c1 = M.ConvLayer(3, 64, activation=M.PARAM_RELU, batch_norm=True, usebias=False)
		# self.c2 = M.ConvLayer(3, 64, activation=M.PARAM_RELU, batch_norm=True, usebias=False)
		# self.c3 = M.ConvLayer(3, 128,activation=M.PARAM_RELU, batch_norm=True, usebias=False)
# 	def forward(self, x):
# 		x = self.c1(x)
# 		x = self.c2(x)
# 		x = self.c3(x)
# 		return x 

def build_cells(step, multiplier, num_chn, in_size, out_size, pp_size, cp=None, cpp=None):
	# pp_size: num below this value have the cpp param, otherwise will be -1
	# cp is the previous channel number. If None, do auto-inference
	cells = []
	connections = []
	for i in range(in_size):
		for j in range(max(0, i-1), min(i+2, out_size)):
			out = num_chn * 2**j
			# prev_prev channel
			if j<pp_size:
				if cpp is None:
					pp = out
				else:
					pp = cpp
			else:
				pp = -1
			# previous channel 
			if cp is None:
				p = num_chn * 2**i
			else:
				p = cp 
			# rate: 0:up 1:same 2:down
			if i==j:
				rate = 1
			elif i<j:
				rate = 2
			else:
				rate = 0
			# build cell
			c = CellBuilder(step, multiplier, pp, p, out, rate)
			cells.append(c)
			connections.append([-1 if pp==-1 else j, i, j])
	return cells, connections


class Body(M.Model):
	def initialize(self, num_layer, num_chn, multiplier, step):
		self.num_layer = num_layer
		self.step = step 

		cp = 128 # channel_previous
		self.c1 = M.ConvLayer(3, 64, stride=2, activation=M.PARAM_RELU, batch_norm=True, usebias=False)
		self.c2 = M.ConvLayer(3, 64, activation=M.PARAM_RELU, batch_norm=True, usebias=False)
		self.c3 = M.ConvLayer(3, 128, stride=2,activation=M.PARAM_RELU, batch_norm=True, usebias=False)

		cells = []
		connections = []

		for i in range(num_layer):
			# pp_size: num below this value have the cpp param, otherwise will be -1
			pp_size = i
			in_size = i+1 if i<4 else 4
			out_size = i+2 if i<2 else 4 
			cp = 128 if i==0 else None
			cpp = 128 if i==1 else None 

			cell, conn = build_cells(step, multiplier, num_chn, in_size, out_size, pp_size, cp, cpp)
			cells.append(cell) 
			connections.append(conn)

		self.cells = cells 
		self.connections = connections

		# for other tasks other than segmentation, we choose other fusing strategy
		# self.aspp4 = ASPP(256, 24)
		# self.aspp8 = ASPP(256, 12)
		# self.aspp16 = ASPP(256, 6)
		# self.aspp32 = ASPP(256, 3)

		self.down1 = FuseDown(3, 128, 256)
		self.down2 = FuseDown(2, 256, 256)
		self.down3 = FuseDown(1, 256, 256)
		self.down4 = M.ConvLayer(1, 256, batch_norm=True, usebias=False)

		self.final_conv = M.ConvLayer(1, 512, batch_norm=True, usebias=False)

	def build(self, inp_shape):
		k = sum(1 for i in range(self.step) for n in range(2+i))
		self.alphas_cell = self.add_variable('alpha_cell', shape=[k, 8], initializer=tf.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal')) # num_type_op = 8
		self.alphas_net = self.add_variable('alpha_net', shape=[self.num_layer, 4, 3], initializer=tf.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal')) # num_type_op = 8
		self.alphas = [self.alphas_cell , self.alphas_net]
		# self.trainable_varaibles = [v for v in self.trainable_varaibles if v not in self.alphas]

	def forward(self, x):
		results = [[],[],[],[]] #4, 8, 16, 32

		x = self.c1(x)
		x = self.c2(x)
		x = self.c3(x)
		results[0].append(x)

		w_cell = tf.nn.softmax(self.alphas_cell, axis=-1)
		w_net = tf.nn.softmax(self.alphas_net, axis=-1)

		for l, (cell, conn) in enumerate(zip(self.cells, self.connections)):
			buff = [[],[],[],[]]
			for ce, con in zip(cell, conn):
				print(con)
				print(len(results[con[0]]))
				pp = None if con[0]==-1 else results[con[0]][-2]
				# if l==1:
				# 	pp = 
				# print(con)
				p = results[con[1]][-1]
				buf = ce(pp, p, w_cell)
				buff[con[2]].append(buf)
			for i,b in enumerate(buff):
				if len(b)>0:
					w_layer = w_net[l,i]
					buff_scale = sum(w_layer[j]*b[j] for j in range(len(b)))
					results[i].append(buff_scale)
					print(len(results[i]))

		asppres4 = self.down1(results[0][-1])
		asppres8 = self.down2(results[1][-1])
		asppres16 = self.down3(results[2][-1])
		asppres32 = self.down4(results[3][-1])

		feat_concat = tf.concat([asppres4, asppres8, asppres16, asppres32], axis=-1)
		feat_last = self.final_conv(feat_concat)
		return feat_last

class FaceRecogNet(M.Model):
	def initialize(self, emb_size, num_class):
		self.body = Body(10,40,5,5)
		self.fc1 = M.Dense(emb_size, usebias=False, batch_norm=True)
		self.classifier = M.MarginalCosineLayer(num_class)
	def forward(self, x, label):
		feat = self.body(x)
		if tf.keras.backend.learning_phase():
			feat = tf.nn.dropout(feat, 0.4)
		feat = M.flatten(feat)
		logits = self.classifier(feat, label, 1.0, 0.5, 0.0)
		logits = logits * 64
		return logits

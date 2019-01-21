import tensorflow as tf 
import layers2 as L 
import modeleag as M 
import numpy as np 
tf.enable_eager_execution()

class res_module(M.Model):
	def initialize(self, out, is_conv4=False):
		self.is_conv4 = is_conv4
		self.bn = L.batch_norm(epsilon=1e-5, is_training=False)
		self.conv1 = L.conv2D(1, out//2)
		self.bn1 = L.batch_norm(epsilon=1e-5, is_training=False)
		self.conv2 = L.conv2D(3, out//2)
		self.bn2 = L.batch_norm(epsilon=1e-5, is_training=False)
		self.conv3 = L.conv2D(1, out)
		if is_conv4:
			self.conv4 = L.conv2D(1, out)

	def forward(self, x):
		residual = x
		x = self.bn(x)
		x = tf.nn.relu(x)
		x = self.conv1(x)
		x = self.bn1(x)
		x = tf.nn.relu(x)
		x = self.conv2(x)
		x = self.bn2(x)
		x = tf.nn.relu(x)
		x = self.conv3(x)
		if self.is_conv4:
			residual = self.conv4(residual)
		return x + residual

class HourGlass(M.Model):
	def initialize(self,n):
		self.n = n
		self.maxpool = L.maxpoolLayer(2)
		self.low1 = []
		for i in range(2):
			self.low1.append(res_module(256))

		if n==1:
			self.low2 = []
			for i in range(2):
				self.low2.append(res_module(256))
		else:
			self.low2 = HourGlass(n-1)

		self.low3 = []
		for i in range(2):
			self.low3.append(res_module(256))

		self.up1 = []
		for i in range(2):
			self.up1.append(res_module(256))


	def forward(self, x):
		size = x.get_shape().as_list()[1]
		low1 = self.maxpool(x)
		for i in range(2):
			low1 = self.low1[i](low1)

		if self.n ==1:
			low2 = low1
			for i in range(2):
				low2 = self.low2[i](low2)
		else:
			low2 = self.low2(low1)

		low3 = low2 
		for i in range(2):
			low3 = self.low3[i](low3)

		# return low1

		up_resize = tf.image.resize_nearest_neighbor(low3, (size, size))
		up_res = x 
		for i in range(2):
			up_res = self.up1[i](up_res)

		return up_res + up_resize


class model(M.Model):
	def initialize(self):
		self.conv1 = L.conv2D(7, 64, pad='VALID', stride=2)
		self.bn1 = L.batch_norm(epsilon=1e-5, is_training=False)
		self.r1 = res_module(128, True)
		self.maxpool = L.maxpoolLayer(2)
		self.r4 = res_module(128)
		self.r5 = res_module(256, True)

		self.hour = []
		for i in range(2):
			self.hour.append(HourGlass(4))

		self.residuals = []
		for i in range(4):
			self.residuals.append(res_module(256))

		self.lin = []
		for i in range(2):
			self.lin.append(L.conv2D(1, 256))
			self.lin.append(L.batch_norm(epsilon=1e-5, is_training=False))

		self.tmpout = []
		self.tmpout.append(L.conv2D(1, 16))
		self.tmpout.append(L.conv2D(1, 16))

		self.ll = []
		self.ll.append(L.conv2D(1, 256))
		self.ll.append(L.conv2D(1, 256))

		self.out_trans = []
		for i in range(2):
			self.out_trans.append(L.conv2D(1,256))


	def forward(self, x):
		# x = tf.pad(x, [[0,0],[3,3],[3,3],[0,0]])
		x = M.pad(x,3)
		x = self.conv1(x)
		x = self.bn1(x)
		x = tf.nn.relu(x)
		x = self.r1(x)
		x = self.maxpool(x)
		x = self.r4(x)
		x = self.r5(x)

		for i in range(2):
			hg = self.hour[i](x)
			ll = hg 
			ll = self.residuals[2*i](ll)
			ll = self.residuals[2*i+1](ll)
			ll = self.lin[2*i](ll)
			ll = self.lin[2*i+1](ll)
			ll = tf.nn.relu(ll)
			tmpout = self.tmpout[i](ll)

			if i==0:
				ll = self.ll[i](ll)
				out_trans = self.out_trans[i](tmpout)
				x = x + ll + out_trans

		return tmpout

mod = model()

saver = M.Saver(mod)
saver.restore('./model/')

x = np.ones((1,256,256,3), dtype=np.float32)
y = mod(x)

# saver.save('./model/hourglass.ckpt')
print(y.shape)
print(y[:,:,:,-1])
# print(np.float32(y).transpose([0,2,3,1]))

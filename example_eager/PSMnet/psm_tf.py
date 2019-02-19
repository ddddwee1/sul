import tensorflow as tf 
import layers2 as L 
L.set_gpu('1')
import modeleag as M 
import numpy as np 

class firstConv(M.Model):
	def initialize(self):
		self.c1 = L.conv2D(3, 32, stride=2, pad='VALID', usebias=False)
		self.bn1 = L.batch_norm(epsilon=1e-5, is_training=False)
		self.c2 = L.conv2D(3, 32, stride=1, pad='VALID', usebias=False)
		self.bn2 = L.batch_norm(epsilon=1e-5, is_training=False)
		self.c3 = L.conv2D(3, 32, stride=1, pad='VALID', usebias=False)
		self.bn3 = L.batch_norm(epsilon=1e-5, is_training=False)

	def forward(self, x):
		x = M.pad(x, 1)
		x = self.c1(x)
		x = self.bn1(x)
		x = tf.nn.relu(x)
		x = M.pad(x, 1)
		x = self.c2(x)
		x = self.bn2(x)
		x = tf.nn.relu(x)
		x = M.pad(x, 1)
		x = self.c3(x)
		x = self.bn3(x)
		x = tf.nn.relu(x)
		return x

class BasicBlock(M.Model):
	def initialize(self, outchn, stride=1, down_sample=False, dilation_rate=1):
		self.dilation_rate = dilation_rate
		self.down_sample = down_sample
		self.c1 = L.conv2D(3, outchn, stride=stride, pad='VALID', dilation_rate=dilation_rate, usebias=False)
		self.bn1 = L.batch_norm(epsilon=1e-5, is_training=False)
		self.c2 = L.conv2D(3, outchn, stride=1, pad='VALID', dilation_rate=dilation_rate, usebias=False)
		self.bn2 = L.batch_norm(epsilon=1e-5, is_training=False)
		if down_sample:
			self.down = L.conv2D(1, outchn, stride=stride, pad='VALID', usebias=False)
			self.bn_down = L.batch_norm(epsilon=1e-5, is_training=False)

	def forward(self, x):
		out = M.pad(x, self.dilation_rate)
		out = self.c1(out)
		out = self.bn1(out)
		out = tf.nn.relu(out)
		out = M.pad(out, self.dilation_rate)
		out = self.c2(out)
		out = self.bn2(out)
		if self.down_sample:
			x = self.down(x)
			x = self.bn_down(x)
		out = out + x
		return out 

class ExtractorBranch(M.Model):
	def initialize(self, pool_size):
		self.c1 = L.conv2D(1, 32, pad='VALID', usebias=False)
		self.bn1 = L.batch_norm(epsilon=1e-5, is_training=False)
		self.avgpool = L.avgpoolLayer(pool_size)

	def forward(self, x):
		x = self.avgpool(x)
		x = self.c1(x)
		x = self.bn1(x)
		x = tf.nn.relu(x)
		return x 

class FeatureExtractor(M.Model):
	def initialize(self):
		self.firstconv = firstConv()
		self.layer1 = [BasicBlock(32) for i in range(3)]
		self.layer2 = [BasicBlock(64, 2 if i==0 else 1, i==0) for i in range(16)]
		self.layer3 = [BasicBlock(128, 1, i==0) for i in range(3)]
		self.layer4 = [BasicBlock(128, 1, False, 2) for i in range(3)]

		self.branch1 = ExtractorBranch(64)
		self.branch2 = ExtractorBranch(32)
		self.branch3 = ExtractorBranch(16)
		self.branch4 = ExtractorBranch(8)

		self.last_c1 = L.conv2D(3, 128, pad='VALID', usebias=False)
		self.last_bn1 = L.batch_norm(epsilon=1e-5, is_training=False)
		self.last_c2 = L.conv2D(1, 32, pad='VALID', usebias=False)

	def forward(self, x):
		x = self.firstconv(x)
		for l in self.layer1:
			x = l(x)
		for l in self.layer2:
			x = l(x)

		output_raw = x

		for l in self.layer3:
			x = l(x)
		for l in self.layer4:
			x = l(x)

		# output_skip = x

		out_b1 = self.branch1(x)
		out_b1 = L.bilinear_upsample(out_b1, 64)
		# out_b1 = tf.pad(out_b1, [[0,0],[1,1],[1,1],[0,0]], mode='symmetric')
		# out_b1 = M.image_transform(out_b1, [1/64., 0., 0.5, 0., 1/64., 0.5, 0., 0.], (x.shape[1], x.shape[2]), interpolation='BILINEAR')

		out_b2 = self.branch2(x)
		out_b2 = L.bilinear_upsample(out_b2, 32)

		out_b3 = self.branch3(x)
		out_b3 = L.bilinear_upsample(out_b3, 16)

		out_b4 = self.branch4(x)
		out_b4 = L.bilinear_upsample(out_b4, 8)

		out_feature = tf.concat([output_raw, x, out_b4, out_b3, out_b2, out_b1], axis=-1)
		out_feature = M.pad(out_feature, 1)
		out_feature = self.last_c1(out_feature)
		out_feature = self.last_bn1(out_feature)
		out_feature = tf.nn.relu(out_feature)
		out_feature = self.last_c2(out_feature)
		return out_feature

class Hourglass(M.Model):
	def initialize(self):
		self.c1 = L.conv3D(3, 64, pad='VALID', stride=2, usebias=False)
		self.bn1 = L.batch_norm(epsilon=1e-5, is_training=False)

		self.c2 = L.conv3D(3, 64, pad='VALID', stride=1, usebias=False )
		self.bn2 = L.batch_norm(epsilon=1e-5, is_training=False)

		self.c3 = L.conv3D(3, 64, pad='VALID', stride=2, usebias=False)
		self.bn3 = L.batch_norm(epsilon=1e-5, is_training=False)

		self.c4 = L.conv3D(3, 64, pad='VALID', stride=1, usebias=False )
		self.bn4 = L.batch_norm(epsilon=1e-5, is_training=False)

		self.c5 = L.deconv3D(3, 64, pad='SAME', stride=2, usebias=False)
		self.bn5 = L.batch_norm(epsilon=1e-5, is_training=False)

		self.c6 = L.deconv3D(3, 32, pad='SAME', stride=2, usebias=False)
		self.bn6 = L.batch_norm(epsilon=1e-5, is_training=False)

	def forward(self, x ,presqu, postsqu):
		x = M.pad3D(x, 1)
		x = self.c1(x)
		x = self.bn1(x)
		x = tf.nn.relu(x)
		out = x

		x = M.pad3D(x, 1)
		x = self.c2(x)
		x = self.bn2(x)
		pre = x 

		if postsqu is not None:
			pre = tf.nn.relu(pre + postsqu)
		else:
			pre = tf.nn.relu(pre)
		x = pre

		x = M.pad3D(x, 1)
		x = self.c3(x)
		x = self.bn3(x)
		x = tf.nn.relu(x)
		x = M.pad3D(x, 1)
		x = self.c4(x)
		x = self.bn4(x)
		x = tf.nn.relu(x)

		out = x
		x = M.pad3D(x, 1)
		x = self.c5(x)
		x = self.bn5(x)
		x = x[:,3:-1,3:-1,3:-1,:]
		if presqu is not None:
			post = tf.nn.relu(x + presqu)
		else:
			post = tf.nn.relu(x + pre)
		
		x = post
		x = M.pad3D(x, 1)
		x = self.c6(x)
		x = self.bn6(x)
		x = x[:,3:-1,3:-1,3:-1,:]
		return x, pre, post

class dres(M.Model):
	def initialize(self):
		self.c1 = L.conv3D(3, 32, pad='VALID', usebias=False)
		self.bn1 = L.batch_norm(epsilon=1e-5, is_training=False)
		self.c2 = L.conv3D(3, 32, pad='VALID', usebias=False)
		self.bn2 = L.batch_norm(epsilon=1e-5, is_training=False)

	def forward(self, x):
		x = M.pad3D(x, 1)
		x = self.c1(x)
		x = self.bn1(x)
		x = tf.nn.relu(x)
		x = M.pad3D(x, 1)
		x = self.c2(x)
		x = self.bn2(x)
		# x = tf.nn.relu(x)
		return x 

class classifier(M.Model):
	def initialize(self):
		self.c1 = L.conv3D(3, 32, pad='VALID', usebias=False)
		self.bn1 = L.batch_norm(epsilon=1e-5, is_training=False)
		self.c2 = L.conv3D(3, 1, pad='VALID', usebias=False)

	def forward(self, x):
		x = M.pad3D(x, 1)
		x = self.c1(x)
		x = self.bn1(x)
		x = tf.nn.relu(x)
		x = M.pad3D(x, 1)
		x = self.c2(x)
		return x

class disparityRegression(M.Model):
	def initialize(self, max_disp):
		disp_value = np.float32(list(range(max_disp))).reshape([1,max_disp,1,1])
		self.disp = tf.get_variable('disp_reg', [1, max_disp, 1, 1] ,initializer=tf.constant_initializer(disp_value),dtype=tf.float32,trainable=False)

	def forward(self, x):
		disp = tf.tile(self.disp, [x.shape[0], 1, x.shape[2], x.shape[3]])
		out = tf.reduce_sum(x*disp, axis=1)
		return out


class PSM(M.Model):
	def initialize(self):
		# self.dres0 = [L.conv3D(3, 32, stride=1, usebias=False, )
		self.feat_ext = FeatureExtractor()
		self.dres0 = dres()
		self.dres1 = dres()
		self.dres2 = Hourglass()
		self.dres3 = Hourglass()
		self.dres4 = Hourglass()

		self.classif1 = classifier()
		self.classif2 = classifier()
		self.classif3 = classifier()

		self.disp = disparityRegression(192)

	def forward(self, imgL, imgR):
		ref_feat = self.feat_ext(imgL)
		tgt_feat = self.feat_ext(imgR)

		cost = []
		for i in range(192//4):
			if i>0:
				cost_buff = tf.concat([ref_feat[:,:,i:,:], tgt_feat[:,:,:-i,:]], axis=-1)
				cost_buff = M.pad(cost_buff, [0,0,i,0])
			else:
				cost_buff = tf.concat([ref_feat, tgt_feat], axis=-1)
			cost.append(cost_buff)
		cost = tf.stack(cost, axis=1)

		cost0 = self.dres0(cost)
		cost0 = tf.nn.relu(cost0)
		cost0 = self.dres1(cost0) + cost0

		out1, pre1, post1 = self.dres2(cost0, None, None)
		out1 = out1 + cost0

		out2, pre2, post2 = self.dres3(out1, pre1, post1)
		out2 = out2 + cost0

		out3, pre3, post3 = self.dres4(out2, pre1, post2)
		out3 = out3 + cost0

		cost1 = self.classif1(out1)
		cost2 = self.classif2(out2) + cost1 
		cost3 = self.classif3(out3) + cost2

		cost3 = L.bilinear_upsample_3d(cost3, 4)
		cost3 = cost3[:,:,:,:,0]
		pred3 = tf.nn.softmax(cost3, axis=1)
		pred3 = self.disp(pred3)
		return pred3


if __name__=='__main__':
	mod = PSM()
	saver = M.Saver(mod)
	saver.restore('./model/')

	img = np.ones([1, 512, 512, 3])
	out = mod(img, img)
	out = out.numpy()
	print(out[0, :, :])
	print(out.shape)

import tensorflow as tf 
import model3 as M 
import numpy as np 
import Inception
import eff_net 

class FaceResNet(M.Model):
	def initialize(self):
		# self.resnet = mobilenet.MobileFaceHead([2, 8, 16, 4])
		# self.resnet = Inception.Inception3(512)
		self.resnet = eff_net.EffNet()

	def forward(self, x):
		feat = self.resnet(x)
		return feat

model = FaceResNet()
a = np.float32(np.ones([1,112,112,3]))
t = model(a)

@tf.function
def go(a):
	_ = model(a)
	run_meta = tf.compat.v1.RunMetadata()
	opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
	flops = tf.compat.v1.profiler.profile(tf.compat.v1.get_default_graph(), run_meta=run_meta, cmd='scope', options=opts)
	print('TF stats gives',flops.total_float_ops)
	return _

a = np.float32(np.ones([1,112,112,3]))
go(a)



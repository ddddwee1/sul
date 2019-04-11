import tensorflow as tf 
import numpy as np 
import model3 as M 

class ArcFace(M.Model):
	def initialize(self, num_classes):
		self.classifier = M.Dense(num_classes, usebias=False, norm=True)
	def forward(self, x):
		x = tf.nn.l2_normalize(x, axis=1)
		x = self.classifier(x)
		return x

class MarginalCosineLayer(M.Model):
	def initialize(self, num_classes):
		self.classifier = M.Dense(num_classes, usebias=False, norm=True)
	def forward(self, x, label, m1=1.0, m2=0.0, m3=0.0):
		# res = cos(m1t + m2) + m3
		# this loss will cause potential unstable
		label = tf.convert_to_tensor(label)
		x = tf.nn.l2_normalize(x, axis=1)
		x = self.classifier(x)
		if not(m1==1.0 and m2==0.0):
			t = tf.gather_nd(x, indices=tf.where(label>0.)) #shape: [N]
			t = tf.math.acos(t)
			### original ###
			# if m1!=1.0:
			# 	t = t*m1
			# if m2!=0.0:
			# 	t = t+m2 
			### end ###
			### experimental: to limit the value not exceed pi ###
			if m1!=1.0:
				t = t*m1
				t1 = t * np.pi / tf.stop_gradient(t)
				t = tf.minimum(t,t1)
			if m2!=0.0:
				t = t+m2 
				t1 = t + np.pi - tf.stop_gradient(t)
				t = tf.minimum(t,t1)
			t = tf.math.cos(t)
			t = tf.expand_dims(t, axis=1)
			x = x*(1-label) + t*label
		x = x - label * m3
		return x

class EnforcedSoftmax(M.Model):
	def initialize(self, num_classes):
		self.classifier = M.Dense(num_classes, usebias=False, norm=True)
	def forward(self, x, label, constrain):
		x = tf.nn.l2_normalize(x, axis=1)
		x = self.classifier(x)
		x = x - tf.abs(x * label * constrain)
		return x 

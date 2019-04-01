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

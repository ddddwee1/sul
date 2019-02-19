import layers as L 
import modeleag as M 
import tensorflow as tf 
import numpy as np 

class GeneratorNet(M.Model):
	def initialize(self):
		self.c1 = M.ConvLayer(5, 32, activation=M.PARAM_RELU, batch_norm=True)
		self.c2 = M.ConvLayer(5, 64, activation=M.PARAM_RELU ,batch_norm=True)
		self.c3 = M.ConvLayer(5, 128, stride=2, activation=M.PARAM_RELU, batch_norm=True)
		self.c4 = M.ConvLayer(5, 256, stride=1, activation=M.PARAM_RELU, batch_norm=True)
		self.c5 = M.ConvLayer(5, 512, stride=2, activation=M.PARAM_RELU, batch_norm=True)
		self.c6 = M.ConvLayer(5, 512, stride=1, activation=M.PARAM_RELU, batch_norm=True)
		self.dc6 = M.DeconvLayer(5, 256, stride=2, activation=M.PARAM_RELU, batch_norm=True)
		self.dc5 = M.DeconvLayer(5, 256, stride=1, activation=M.PARAM_RELU, batch_norm=True)
		self.dc4 = M.DeconvLayer(5, 256, stride=2, activation=M.PARAM_RELU, batch_norm=True)
		self.dc3 = M.DeconvLayer(5, 128, stride=1, activation=M.PARAM_RELU, batch_norm=True)
		self.dc2 = M.DeconvLayer(5, 64, stride=1, activation=M.PARAM_RELU, batch_norm=True)
		self.dc1 = M.DeconvLayer(5, 1, stride=1, activation=M.PARAM_TANH)

	def forward(self,x):
		x = self.c1(x)
		x = self.c2(x)
		x = self.c3(x)
		layer3 =x 
		x = self.c4(x)
		x = self.c5(x)
		x = self.c6(x)
		x = self.dc6(x)
		x = self.dc5(x)
		x = self.dc4(x)
		x = self.dc3(x)
		x = self.dc2(x)
		x = self.dc1(x)
		return x, layer3 

class DecoderNet(M.Model):
	def initialize(self):
		self.dc1 = self.DeconvLayer(5, 128, activation=M.PARAM_RELU, batch_norm=True)
		self.dc2 = self.DeconvLayer(5, 128, stride=2)

class DisNet(M.Model):
	def initialize(self):
		# input 256
		self.c1 = M.ConvLayer(5, 32, activation=M.PARAM_RELU, batch_norm=True)
		self.c2 = M.ConvLayer(5, 64, stride=2 ,activation=M.PARAM_RELU, batch_norm=True) # 128
		self.c3 = M.ConvLayer(5, 128, stride=2, activation=M.PARAM_RELU, batch_norm=True) # 64
		self.c4 = M.ConvLayer(5, 256, stride=2, activation=M.PARAM_RELU, batch_norm=True) # 32

	def forward(self,x): 
		x = self.c1(x)
		x = self.c2(x)
		x = self.c3(x)
		x = self.c4(x)
		return x

day_night = GeneratorNet()
night_day = GeneratorNet()

day_dis = DisNet()
night_dis = DisNet()

def get_day(img, model):
	return model(img)

def get_night(img, model):
	return model(img)

def get_gan_loss(dis_real, dis_fake):
	loss_d_real = tf.reduce_mean(tf.square(dis_real - tf.ones_like(dis_real)))
	loss_d_fake = tf.reduce_mean(tf.square(dis_fake - tf.zeros_like(dis_fake)))
	loss_d = 0.5 * (loss_d_real + loss_d_fake)
	loss_g = tf.reduce_mean(tf.square(dis_fake - tf.ones_like(dis_fake)))
	return loss_d, loss_g

def consist_loss(img, fake):
	return tf.reduce_mean(tf.abs(img - fake))

def pao(day, night, models):
	with tf.GradientTape() as tape:
		day_night, night_day, day_dis, night_dis = models

		gen_night1 = day_night(day)
		gen_day1 = night_day(gen_night1)

		gen_day2 = night_day(night)
		gen_night2 = day_night(gen_day2)

		dis_real_day = day_dis(day)
		dis_fake_day = day_dis(gen_day1)
		dis_fake_day2 = day_dis(gen_day2)

		dis_real_night = night_dis(night)
		dis_fake_night = night_dis(gen_night1)
		dis_fake_night2 = night_dis(gen_night2)

		loss_d_day1, loss_g_day1 = get_gan_loss(dis_real_day, dis_fake_day)
		loss_d_day2, loss_g_day2 = get_gan_loss(dis_real_day, dis_fake_day2)

		loss_d_day, loss_g_day = loss_d_day1+ loss_d_day2, loss_g_day1 + loss_g_day2

		loss_d_night1, loss_g_night1 = get_gan_loss(dis_real_night, dis_fake_night)
		loss_d_night2, loss_g_night2 = get_gan_loss(dis_real_night, dis_fake_night2)
		loss_d_night, loss_g_night = loss_d_night1+loss_d_night2, loss_g_night1+loss_g_night2

		const_day = consist_loss(day - gen_day1)
		const_night = consist_loss(night - gen_night2)

	return [loss_d_day, loss_g_day, loss_d_night, loss_g_night, const_day, const_night], tape

def gengxin(losses, model, optim, tape):
	day_night, night_day, day_dis, night_dis = models

	generator_vars = day_night.variables + night_day.variables

	variables = [day_dis.variables, night_day.variables, night_dis.variables, day_night.variables, generator_vars, generator_vars]
	grads = tape.gradient(losses, variables)

	for g,v in zip(grads, variables):
		optim.apply_gradients(M.zip_grads(g,v))


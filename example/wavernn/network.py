import numpy as np 
import tensorflow as tf 
import model3 as M 
from tensorflow.keras.layers import GRU, Bidirectional, LSTMCell, GRUCell

class ResBlock(M.Model):
	def initialize(self, chn):
		self.c1 = M.ConvLayer1D(1, chn, activation=M.PARAM_RELU, batch_norm=True, usebias=False)
		self.c2 = M.ConvLayer1D(1, chn, activation=M.PARAM_RELU, batch_norm=True, usebias=False)
	def forward(self, x):
		residual = x 
		x = self.c1(x)
		x = self.c2(x)
		return x + residual

class MelResNet(M.Model):
	def initialize(self, num_blocks, chn1, outchn, pad):
		kszie = pad * 2 + 1 
		self.cin = M.ConvLayer1D(kszie, chn1, activation=M.PARAM_RELU, batch_norm=True, usebias=False)
		self.layers = []
		for i in range(num_blocks):
			self.layers.append(ResBlock(chn1))
		self.cout = M.ConvLayer1D(1, outchn)
	def forward(self, x):
		x = self.cin(x)
		for c in self.layers: x = c(x)
		x = self.cout(x)
		return x 

class Stretch2D(M.Model):
	def initialize(self, xscale, yscale):
		self.xscale = xscale
		self.yscale = yscale
	def forward(self, x):
		x = tf.tile(x, [1,self.yscale,self.xscale,1])
		return x 

class UpsampleNet(M.Model):
	def initialize(self, upscales, chn1, num_blocks, outchn, pad):
		totalscale = np.cumproduct(upscales)[-1]
		self.indent = pad * totalscale
		self.resnet = MelResNet(num_blocks, chn1, outchn, pad)
		self.stretch = Stretch2D(totalscale, 1)
		self.uplayers = []
		for s in upscales:
			kszie = [1, s*2+1]
			stretch = Stretch2D(s, 1)
			c = M.ConvLayer(kszie, 1, usebias=False)
			self.uplayers += [stretch, c]
	def forward(self, x):
		aux = self.resnet(x)
		aux = self.stretch(tf.expand_dims(aux,axis=-1))
		aux = tf.squeeze(aux, -1)
		x = tf.expand_dims(x,-1)
		for c in self.uplayers: x = c(x)
		x = tf.squeeze(x, -1)[:,:,self.indent:-self.indent]
		return tf.transpose(x, [0,2,1]), tf.transpose(aux, [0,2,1])

class WaveRNN(M.Model):
	def initialize(self, rnndim, fcdim, bits, pad, upscales, featdims, chn1, outchn, num_blocks, hop_length, samplerate):
		self.n_classes = 30
		self.rnndim = rnndim
		self.auxdim = outchn // 4
		self.hop_length = hop_length
		self.samplerate = samplerate

		self.upsample = UpsampleNet(upscales, chn1, num_blocks, outchn, pad)
		self.I = M.Dense(rnndim)
		self.rnn1 = GRU(rnndim, return_sequences=True)
		self.rnn2 = GRU(rnndim, return_sequences=True)
		self.fc1 = M.Dense(fcdim, activation=M.PARAM_RELU)
		self.fc2 = M.Dense(fcdim, activation=M.PARAM_RELU)
		self.fc3 = M.Dense(self.n_classes)

		self.step = 0

	def forward(self, x, mels):
		self.step += 1
		bsize = x.shape[0]

		mels, aux = self.upsample(mels)
		aux_idx = [self.auxdim * i for i in range(5)]
		a1 = aux[:,:,aux_idx[0]:aux_idx[1]]
		a2 = aux[:,:,aux_idx[1]:aux_idx[2]]
		a3 = aux[:,:,aux_idx[2]:aux_idx[3]]
		a4 = aux[:,:,aux_idx[3]:aux_idx[4]]

		x = tf.concat([tf.expand_dims(x,-1), mels, a1], axis=2)
		x = self.I(x)
		res = x 
		x = self.rnn1(x)
		x = x + res 

		res = x 
		x = tf.concat([x,a2], axis=2)
		x = self.rnn2(x)
		x = x + res 

		x = tf.concat([x,a3], axis=2)
		x = self.fc1(x)
		x = tf.concat([x,a4], axis=2)
		x = self.fc2(x)

		x = self.fc3(x)
		return x 



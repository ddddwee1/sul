import numpy as np 
import tensorflow as tf 
import model3 as M 
import config 
import util 
from tensorflow.python.training.tracking.data_structures import NoDependency

LSTM_DIM = config.LSTM_DIM
IN_DIM = config.IN_DIM

class Decoder(M.Model):
	def initialize(self):
		self.c1 = M.Dense(512, activation=M.PARAM_LRELU)
		self.c2 = M.Dense(256, activation=M.PARAM_LRELU)
		self.c3 = M.Dense(IN_DIM)
	def forward(self, x):
		return self.c3(self.c2(self.c1(x)))

class LSTM2(M.Model):
	def initialize(self, outdim):
		self.hc = NoDependency({})
		self.outdim = outdim
		self.LSTMCell = M.LSTMCell(outdim)
	def forward(self, x, branch, init_hc=False):
		branch = str(branch)
		if (not (branch in self.hc)) or (init_hc):
			self.hc[branch] = [tf.zeros([x.shape[0], self.outdim]), tf.zeros([x.shape[0], self.outdim])]
		h,c = self.hc[branch]
		# print('h',h.shape,'c',c.shape,'x',x.shape)
		next_h, next_c = self.LSTMCell(x, h, c)
		self.hc[branch] = [next_h, next_c]
		return next_h


class PosePredNet(M.Model):
	def initialize(self):
		self.lstm_l1 = LSTM2(LSTM_DIM)
		self.lstm_l2 = LSTM2(LSTM_DIM)
		self.dec = Decoder()

	def forward(self, enc_in, pred_step):
		vs = util.convert_velocity(enc_in)
		out_layer1 = [self.lstm_l1(vs[i], 0, init_hc=i==0) for i in range(len(vs))]
		out_layer2a = [self.lstm_l2(x, 0, init_hc=i==0) for i,x in enumerate(out_layer1[0::2])]
		out_layer2b = [self.lstm_l2(x, 1, init_hc=i==0) for i,x in enumerate(out_layer1[1::2])]
		out_layer2 = [out_layer2a, out_layer2b]

		predictions = []
		pred_pos = tf.convert_to_tensor(enc_in[-1])

		for i in range(pred_step):
			step = len(vs) + i - 1
			pred_v = self.dec(tf.concat([out_layer1[step], out_layer2[step%2][step//2]], axis=1))
			pred_pos = pred_pos + pred_v
			predictions.append(pred_pos)
			if i!=(pred_step-1):
				out_layer1.append(self.lstm_l1(pred_v, 0))
				step += 1 
				# print(step, len(out_layer1))
				out_layer2[step%2].append(self.lstm_l2(out_layer1[step], step%2))
		return predictions

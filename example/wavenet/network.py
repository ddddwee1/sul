import tensorflow as tf 
import model3 as M 
from tensorflow.keras.layers import GRU, Bidirectional, LSTMCell, GRUCell
import time 

class HighwayNet(M.Model):
	def initialize(self, size):
		self.fc1 = M.Dense(size, activation=M.PARAM_RELU)
		self.fc2 = M.Dense(size, activation=M.PARAM_SIGMOID)
	def forward(self, x):
		# self-attention
		x1 = self.fc1(x)
		g = self.fc2(x)
		y = g * x1 + (1 - g) * x 
		return y 

class PreNet(M.Model):
	def initialize(self, out1=256, out2=128, dropout=0.5):
		self.fc1 = M.Dense(out1, activation=M.PARAM_RELU)
		self.fc2 = M.Dense(out2, activation=M.PARAM_RELU)
		self.dropout = dropout
	def forward(self, x):
		x = self.fc1(x)
		if tf.keras.backend.learning_phase():
			x = tf.nn.dropout(x, self.dropout)
		x = self.fc2(x)
		if tf.keras.backend.learning_phase():
			x = tf.nn.dropout(x, self.dropout)
		return x 

class CBHG(M.Model):
	def initialize(self, K, channels, proj_chn, num_highway):
		self.kernels = [i for i in range(1, K+1)]
		self.conv1d_bank = []
		for k in self.kernels:
			conv = M.ConvLayer1D(k, channels, batch_norm=True, usebias=False, activation=M.PARAM_RELU)
			self.conv1d_bank.append(conv)
		self.maxpool = M.MaxPool(2, 1) # implement this later
		
		self.conv_proj1 = M.ConvLayer1D(3, proj_chn[0], batch_norm=True, usebias=False, activation=M.PARAM_RELU)
		self.conv_proj2 = M.ConvLayer1D(3, proj_chn[1], batch_norm=True, usebias=False)

		if proj_chn[-1] != channels:
			self.pre_highway = M.ConvLayer1D(1, channels, usebias=False)

		self.highways = []
		for i in range(num_highway):
			hw = HighwayNet(channels)
			self.highways.append(hw)

		self.rnn = Bidirectional(GRU(channels,return_sequences=True)) # implement this later

	def forward(self, x):
		# print('CBHGIN',x.shape)
		residual = x 
		seqlen = x.shape[1]
		convbank = []
		for conv in self.conv1d_bank:
			c = conv(x)
			convbank.append(c[:,:seqlen])
		x = tf.concat(convbank, axis=-1)

		x = self.maxpool(x)[:,:seqlen]

		x = self.conv_proj1(x)
		x = self.conv_proj2(x)

		x = x + residual

		print('XSHAPE',x.shape)

		if hasattr(self, 'pre_highway'):
			x = self.pre_highway(x)
		for hw in self.highways:
			x = hw(x)

		x = self.rnn(x)
		return x 

class Attention(M.Model):
	def initialize(self, dim):
		self.W = M.Dense(dim, usebias=False)
		self.v = M.Dense(1, usebias=False)
	def forward(self, enc_seq, query, t):
		q_proj = tf.expand_dims(self.W(query),axis = 1)
		u = self.v(tf.tanh(q_proj + enc_seq))
		scr = tf.nn.softmax(u, axis=1)
		return scr 

class LSA(M.Model):
	def initialize(self, att_dim, kszie=31, filters=32):
		self.conv = M.ConvLayer1D(kszie, filters)
		self.L = M.Dense(att_dim)
		self.W = M.Dense(att_dim)
		self.v = M.Dense(1, usebias=False)
		self.cumulate = None
		self.att = None 

	def init_att(self, enc_seq):
		b, t, c = enc_seq.shape 
		self.cumulate = tf.zeros([b,t])
		self.att = tf.zeros([b,t])

	def forward(self, enc_seq, query, t):
		if t==0:
			self.init_att(enc_seq)
		q_proj = self.W(query)
		q_proj = tf.expand_dims(q_proj, axis=1)
		loc = tf.stack([self.cumulate, self.att], axis=-1)
		loc_proj = self.L(self.conv(loc))
		u = self.v(tf.tanh(q_proj + loc_proj + enc_seq))
		u = tf.squeeze(u, axis=-1)
		scr = tf.sigmoid(u)
		scr = u / tf.reduce_sum(u, axis=1, keepdims=True)
		self.att = scr
		self.cumulate = self.cumulate + scr 
		scr = tf.expand_dims(scr, axis=-1)
		return scr 

class Encoder(M.Model):
	def initialize(self, embed_dim, num_chars, cbhg_chn, K, num_highway, dropout):
		self.num_chars = num_chars
		self.embedding = M.Dense(embed_dim)
		self.prenet = PreNet()
		self.cbhg = CBHG(K=K, channels=cbhg_chn, proj_chn=[cbhg_chn, cbhg_chn], num_highway=num_highway)

	def forward(self, x):
		x = tf.one_hot(x, depth=self.num_chars)
		x = self.embedding(x)
		x = self.prenet(x)
		x = self.cbhg(x)
		# print('ENC',x.shape)
		return x 

class Decoder(M.Model):
	def initialize(self, n_mels, dec_dim, lstm_dim):
		self.max_r = 20 
		self.r = None 
		self.n_mels = n_mels
		self.prenet = PreNet()
		self.att_net = LSA(dec_dim)
		self.att_rnn = GRUCell(dec_dim)
		self.rnn_inp = M.Dense(lstm_dim)
		self.res_rnn1 = LSTMCell(lstm_dim)
		self.res_rnn2 = LSTMCell(lstm_dim)
		self.mel_proj = M.Dense(n_mels * self.max_r, usebias=False)

	def zoneout(self, prev, curr, rate=0.1):
		mask = tf.random.uniform(prev.shape) + 1 - rate
		mask = tf.floor(mask)
		return prev * mask + (1 - mask) * curr

	def forward(self, enc_seq, enc_seq_proj, prenet_in, hid_states, cell_states, context_vec, t):
		enc_shape = enc_seq.shape 
		attn_hid, rnn1_hid, rnn2_hid = hid_states
		rnn1_cell, rnn2_cell = cell_states
		pre_out = self.prenet(prenet_in)
		attn_rnn_in = tf.concat([context_vec, pre_out], axis=-1)

		# print(attn_rnn_in.shape)
		# next attention hidden state
		attn_hid = self.att_rnn(attn_rnn_in, [attn_hid])[0]

		scrs = self.att_net(enc_seq_proj, attn_hid, t)

		context_vec = tf.transpose(scrs, [0,2,1]) @ enc_seq
		context_vec = tf.squeeze(context_vec, axis=1)

		x = tf.concat([context_vec, attn_hid], axis=1)
		x = self.rnn_inp(x)
		# print('XSHAPE',x.shape)
		_, (rnn1_hid2, rnn1_cell) = self.res_rnn1(x, (rnn1_hid, rnn1_cell))

		if tf.keras.backend.learning_phase():
			rnn1_hid = self.zoneout(rnn1_hid, rnn1_hid2)
		else:
			rnn1_hid = rnn1_hid2
		x = x + rnn1_hid

		_, (rnn2_hid2, rnn2_cell) = self.res_rnn2(x, (rnn2_hid, rnn2_cell))
		if tf.keras.backend.learning_phase():
			rnn2_hid = self.zoneout(rnn2_hid, rnn2_hid2)
		else:
			rnn2_hid = rnn2_hid2
		x = x + rnn2_hid

		mels = self.mel_proj(x)
		mels = tf.reshape(mels, [enc_shape[0], self.n_mels, self.max_r])[:,:,:self.r]

		hid_states = (attn_hid, rnn1_hid, rnn2_hid)
		cell_states = (rnn1_cell, rnn2_cell)
		return mels, scrs, hid_states, cell_states, context_vec

class Tacotron(M.Model):
	def initialize(self, embed_dim, num_chars, enc_dim, dec_dim, n_mels, fft_bins, postnet_dim,
					enc_K, lstm_dim, postnet_K, num_highway, dropout):
		self.n_mels = n_mels
		self.lstm_dim = lstm_dim
		self.dec_dim = dec_dim
		self.encoder = Encoder(embed_dim, num_chars, enc_dim, enc_K, num_highway, dropout)
		self.enc_proj = M.Dense(dec_dim, usebias=False)
		self.decoder = Decoder(n_mels, dec_dim, lstm_dim)
		self.postnet = CBHG(postnet_K, postnet_dim, [256, 80], num_highway)
		self.post_proj = M.Dense(fft_bins, usebias=False)

	def set_r(self, r):
		print('SET R:',r)
		self.decoder.r = r 
		self.r = r 

	def forward(self, x, m):
		m = tf.convert_to_tensor(m)
		bsize, _, step = m.shape 

		attn_hid = tf.zeros([bsize, self.dec_dim])
		rnn1_hid = tf.zeros([bsize, self.lstm_dim])
		rnn2_hid = tf.zeros([bsize, self.lstm_dim])
		hid_states = (attn_hid, rnn1_hid, rnn2_hid)
		rnn1_cell = tf.zeros([bsize, self.lstm_dim])
		rnn2_cell = tf.zeros([bsize, self.lstm_dim])
		cell_states = (rnn1_cell, rnn2_cell)

		go_frame = tf.zeros([bsize, self.n_mels]) # starter

		context_vec = tf.zeros([bsize, self.dec_dim])

		# print(x.shape)
		t1 = time.time()
		enc_seq = self.encoder(x)
		# print('abc',enc_seq.shape)
		t2 = time.time()
		enc_seq_proj = self.enc_proj(enc_seq)
		# print('abc',enc_seq_proj.shape)

		mel_outs, attn_scrs = [] , []
		t3 = time.time()
		for t in range(0, step, self.r):
			prenet_in = m[:,:,t-1] if t>0 else go_frame
			mels, scrs, hid_states, cell_states, context_vec = self.decoder(enc_seq, enc_seq_proj, 
																			prenet_in, hid_states, cell_states, context_vec, t)
			mel_outs.append(mels)
			attn_scrs.append(scrs)
		mel_outs = tf.concat(mel_outs, axis=2)
		mel_outs_t = tf.transpose(mel_outs, [0,2,1])
		t4 = time.time()

		# print('MEL_OUTS', mel_outs.shape)
		post_out = self.postnet(mel_outs_t)
		linear = self.post_proj(post_out)
		linear = tf.transpose(linear, [0,2,1])
		t5 = time.time()

		print('T', t2-t1, t3-t2, t4-t3, t5-t4)

		# print('FINAL_OUTS:', mel_outs.shape, linear.shape)
		# input('LINE248PAUSE')

		# useless for me
		attn_scrs = tf.concat(attn_scrs, axis=1)

		return mel_outs, linear, attn_scrs

	def generate(self, x, steps=20000):
		tf.keras.backend.set_learning_phase(False)
		x = tf.convert_to_tensor(x)
		bsize = 1 
		x = tf.expand_dims(x, axis=0)
		attn_hid = tf.zeros([bsize, self.dec_dim])
		rnn1_hid = tf.zeros([bsize, self.lstm_dim])
		rnn2_hid = tf.zeros([bsize, self.lstm_dim])
		hid_states = (attn_hid, rnn1_hid, rnn2_hid)
		rnn1_cell = tf.zeros([bsize, self.lstm_dim])
		rnn2_cell = tf.zeros([bsize, self.lstm_dim])
		cell_states = (rnn1_cell, rnn2_cell)

		go_frame = tf.zeros([bsize, self.n_mels]) # starter

		context_vec = tf.zeros([bsize, self.dec_dim])

		enc_seq = self.encoder(x)
		enc_seq_proj = self.enc_proj(enc_seq)

		mel_outs, attn_scrs = [] , []
		for t in range(0, step, self.r):
			prenet_in = mel_outs[-1][:,:,-1] if t>0 else go_frame
			mels, scrs, hid_states, cell_states, context_vec = self.decoder(enc_seq, enc_seq_proj, 
																			prenet_in, hid_states, hid_states, cell_states, context_vec, t)
			mel_outs.append(mels)
			attn_scrs.append(scrs)
			if (mels < -3.8).all() and t > 10: break
		mel_outs = tf.concat(mel_outs, axis=2)

		post_out = self.postnet(mel_outs)
		linear = self.post_proj(post_out)
		linear = tf.transpose(linear, [0,2,1])

		return mel_outs, linear, None

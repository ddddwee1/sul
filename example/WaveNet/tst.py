import tensorflow as tf 
import numpy as np 
import model3 as M 
import network 
import util 
import config 
import datasets
from text.symbols import symbols

if __name__=='__main__':
	model = network.Tacotron(embed_dim=config.tts_embed_dims,
							num_chars=len(symbols),
							enc_dim=config.tts_encoder_dims,
							dec_dim=config.tts_decoder_dims,
							n_mels=config.num_mels,
							fft_bins=config.num_mels,
							postnet_dim=config.tts_postnet_dims,
							enc_K=config.tts_encoder_K,
							lstm_dim=config.tts_lstm_dims,
							postnet_K=config.tts_postnet_K,
							num_highway=config.tts_num_highways,
							dropout=config.tts_dropout)
	model.set_r(1)

	trainset = datasets.get_tts_dataset(config.data_path, 8, 1)
	x, m, ids, _ = trainset.get_next() 
	# m1_hat, m2_hat, att = model(x, m)
	# print(m1_hat.shape)
	# print(m2_hat.shape)
	# print(m.shape)

	with tf.GradientTape() as tape:
		debug = model(x,m)
		loss = tf.reduce_mean(tf.abs(debug - tf.zeros_like(debug)))
	print(debug.shape)
	
	optim = tf.optimizers.Adam(0.0001)
	grads = tape.gradient(loss, model.trainable_variables)
	optim.apply_gradients(zip(grads, model.trainable_variables))
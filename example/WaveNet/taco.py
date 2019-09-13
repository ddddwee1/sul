import tensorflow as tf 
import numpy as np 
import model3 as M 
import network 
import util 
import config 
import datasets
from text.symbols import symbols
from tqdm import tqdm
import time 

def create_gta(model, trainset, savepath):
	for i in range(trainset.maxiters):
		x, m, ids, m_lens = trainset.get_next()
		_, gta, _ = model(x, m)
		gta = gta.numpy()
		for j in range(len(ids)):
			mel = gta[j][:,:m_lens[j]]
			mel = (mel + 4) / 8
			ID = ids[j]
			np.save(f'{savepath}{ID}.npy', mel, allow_pickle=False)
		print(f'{i}/{iters}')


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

saver = M.Saver(model)
saver.restore(config.model_folder)


# @tf.function
def grad_loss(x,m):
	with tf.GradientTape() as tape:
		m1_hat, m2_hat, att = model(x, m)
		m1_ls = tf.reduce_mean(tf.abs(m1_hat - m))
		m2_ls = tf.reduce_mean(tf.abs(m2_hat - m))
		loss = m1_ls + m2_ls
	t1 = time.time()
	grads = tape.gradient(loss, model.trainable_variables)
	t2 = time.time()
	print('Time grad:', t2-t1)
	return grads, loss

def tts_train_loop(optim, trainset, epochs):
	for e in range(epochs):
		bar = tqdm(range(trainset.maxiters))
		for i in bar:
			x, m, ids, _ = trainset.get_next() 
			grads, loss = grad_loss(x,m)
			optim.apply_gradients(zip(grads, model.trainable_variables))

			lsmsg = 'Epoch:%d Iter:%d Loss:%.4f'%(e, i, loss)
			bar.set_description(lsmsg)
			# print()

for r,lr,maxepoch,bsize in config.schedule:
	optim = tf.optimizers.Adam(lr)
	model.set_r(r)
	trainset = datasets.get_tts_dataset(config.data_path, bsize, r)

	tts_train_loop(optim, trainset, maxepoch)

print('Training complete')
print('Create GroundTruth Aligned Dataset..')
trainset = datasets.get_tts_dataset(config.data_path, 8, model.get_r())
create_gta(model, trainset, config.gta_path)

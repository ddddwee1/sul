import tensorflow as tf 
import numpy as np 
import model3 as M 
import network 
import util 
import config 

def tts_train_loop(model, optim, trainset, epochs):
	for e in range(epochs):
		for i in range(trainset.maxiters):
			x, m, ids, _ = trainset.get_next() 
			with tf.GradientTape() as tape:
				m1_hat, m2_hat, att = model(x, m)
				m1_ls = tf.abs(m1_hat - m)
				m2_ls = tf.abs(m2_hat - m)
				loss = m1_ls + m2_ls
			grads = tape.gradient(loss, model.trainable_variables)
			optim.apply_gradients(zip(grads, model.trainable_variables))

			print('Epoch:%d\tIter:%d\tLoss:%.4f'%(e, i, loss))

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

if __name__=='__main__':
	model = network.Taco()

	saver = M.Saver(model)
	saver.restore(config.model_folder)

	
	for r,lr,maxepoch,bsize in config.schedule:
		optim = tf.optimizers.Adam(lr)
		model.set_r(r)
		trainset, _ = util.get_tts_dataset(config.datapath, bsize, r)

		tts_train_loop(model, optim, trainset, maxepoch)

	print('Training complete')
	print('Create GroundTruth Aligned Dataset..')
	trainset, _ = util.get_tts_dataset(config.datapath, 8, model.get_r())
	create_gta(model, trainset, config.gta_path)

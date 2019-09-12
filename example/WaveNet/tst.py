import tensorflow as tf 
import numpy as np 
import model3 as M 
import network 
import util 
import config 

if __name__=='__main__':
	model = network.Tacotron()
	trainset, _ = util.get_tts_dataset(config.datapath, bsize, r)
	x, m, ids, _ = trainset.get_next() 
	m1_hat, m2_hat, att = model(x, m)
	print(m1_hat.shape)
	print(m2_hat.shape)
	
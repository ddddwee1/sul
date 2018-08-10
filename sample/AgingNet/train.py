import AIM
import tensorflow as tf 
import numpy as np 
import model as M 

# get data_reader
import data_reader
data_reader = data_reader.data_reader('outpt.txt')

M.set_gpu('1')

BSIZE = 32
ITER_PER_EPOC = 200000//BSIZE
EPOC = 30
MAX_ITER = ITER_PER_EPOC * EPOC

aim_mod = AIM.AIM(data_reader.age_class, data_reader.max_id)

ETA = M.ETA(MAX_ITER)
ETA.start()
for iteration in range(MAX_ITER+1):
	img, target, uniform, age, idn = data_reader.get_train_batch(BSIZE)
	losses, generated = aim_mod.train(img, target, uniform, age, idn,normalize=True)
	if iteration%10==0:
		print('------ Iteration %d ---------'%iteration)
		aim_mod.display_losses(losses)
		print('ETA',ETA.get_ETA(iteration))
	if iteration%1000==0 and iteration>0:
		aim_mod.save('%d.ckpt'%iteration)

import AIM_gen
import tensorflow as tf 
import numpy as np 
import model as M 
import cv2

M.set_gpu('3')

aim_mod = AIM_gen.AIM_gen(7)

# get data_reader
import data_reader
data_reader = data_reader.data_reader('outpt.txt')

BSIZE = 32
ITER_PER_EPOC = 200000//BSIZE
EPOC = 50
MAX_ITER = ITER_PER_EPOC * EPOC

ETA = M.ETA(MAX_ITER)
ETA.start()
for iteration in range(MAX_ITER+1):
	img, age_fake, age = data_reader.get_train_batch(BSIZE)
	losses = aim_mod.train(img, age_fake, age,normalize=True)
	if iteration%10==0:
		print('------ Iteration %d ---------'%iteration)
		aim_mod.display_losses(losses)
		print('ETA',ETA.get_ETA(iteration))
	if iteration%1000==0 and iteration>0:
		aim_mod.save('%d.ckpt'%iteration)

	if iteration%100==0:

		gen = aim_mod.eval(img,age_fake)
		img = np.uint8(img)
		gen = np.uint8(gen)
		for i in range(BSIZE):
			cv2.imwrite('./res/%d_%d_1.jpg'%(iteration,i),img[i])
			cv2.imwrite('./res/%d_%d_2.jpg'%(iteration,i),gen[i])

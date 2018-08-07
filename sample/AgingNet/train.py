import AIM
import tensorflow as tf 
import numpy as np 
import model as M 

M.set_gpu('1')

MAX_ITER = 10000

aim_mod = AIM.AIM(512,8)
ETA = M.ETA(MAX_ITER)

ETA.start()
for iteration in range(MAX_ITER+1):
	img = np.random.random([16,128,128,3])
	target = np.random.random([16,128,128,3])
	uniform = np.random.random([16,2,2,512])
	age = np.zeros([16,8])
	age[:,2] = 0.9
	idn = np.zeros([16,512])
	idn[:,2] = 0.9

	losses, generated = aim_mod.train(img, target, uniform, age, idn,normalize=False)
	if iteration%10==0:
		print('------ Iteration %d ---------'%iteration)
		aim_mod.display_losses(losses)
		print('ETA',ETA.get_ETA(iteration))
	if iteration%1000==0:
		aim_mod.save('%d.ckpt'%iteration)
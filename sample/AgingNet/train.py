import AIM
import tensorflow as tf 
import numpy as np 

aim_mod = AIM.AIM(512,8)

# train(self, img, target, uniform, age_lab, id_lab, normalize=True)

img = np.random.random([4,128,128,3])
target = np.random.random([4,128,128,3])
uniform = np.random.random([4,2,2,512])
age = np.zeros([4,8])
age[:,2] = 0.9
idn = np.zeros([4,512])
idn[:,2] = 0.9

losses, generated = aim_mod.train(img, target, uniform, age, idn,normalize=False)
aim_mod.display_losses(losses)
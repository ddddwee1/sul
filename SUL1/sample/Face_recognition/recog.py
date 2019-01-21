import resnet3 as net 
import numpy as np 
import cv2 


img = cv2.imread('9_1.jpg')
res1 = net.eval(img.reshape([-1,128,128,3]))
img = cv2.imread('10_1.jpg')
res2 = net.eval(img.reshape([-1,128,128,3]))

res1 = res1/np.linalg.norm(res1)
res2 = res2/np.linalg.norm(res2)

cosres = res1*res2
cosres = cosres.sum()

print('Sim1: ',cosres)
print('Sim2: ',np.arctan(cosres))
import numpy as np 
import cv2 

# fname, label
f = open('train_list.txt')
for i in f:
	i = i.strip()
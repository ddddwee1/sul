import tensorflow as tf 
import numpy as np 
import data_reader
import network 

VALIDATION_RATIO = 0.2
ITERATIONS = 100000
BATCH_SIZE = 8
BATCH_SIZE_VAL = 16
VALIDATE_INTERVAL = 10000
SAVE_INTERVAL = 10000

reader = data_reader.data_reader(VALIDATION_RATIO)
net = network.text_net()

for i in range(ITERATIONS+1):
	image_batch, mask_batch, geo_batch, rot_batch, quad_batch = reader.get_training_batch(BATCH_SIZE)
	losses = net.train(image_batch, mask_batch, geo_batch, rot_batch, quad_batch)
	if i%10==0:
		# print loss
		net.print_loss(losses)
	if i%VALIDATE_INTERVAL==0:
		image_batch, mask_batch, geo_batch, rot_batch, quad_batch = reader.get_training_batch(BATCH_SIZE_VAL)
		losses = net.val(image_batch, mask_batch, geo_batch, rot_batch, quad_batch)
		net.print_loss(losses)
	if i%SAVE_INTERVAL==0 and i>0:
		net.save('%d.ckpt'%i)

import network 
import modeleag as M 
import numpy as np 
import tensorflow as tf 

from data_reader import *

BSIZE = 16
ITER = 100001
SAVE_INTERVAL = 2000

txt_fn='./task1/train/gt/'
def load_dataset(txt_fn):
	result = data_reader(txt_fn)
	return result

def processed_bundle(bundle):
	img, annot = bundle
	img, annot = process_image(img, annot)
	img, conf, xywh = annot_to_grid(img, annot, 16)
	return bundle

def post_process(bundle):
	return list(zip(*bundle))
	
if __name__=='__main__':
	net = network.network()
	optim = tf.train.AdamOptimizer(0.0001)
	saver = M.Saver(net, optim)
	saver.restore('./model/')

	train_data = load_dataset(txt_fn)
	print(len(train_data))

	train_loader = M.DataReader(data=train_data, fn=processed_bundle, batch_size=BSIZE, \
		shuffle=True, random_sample=True, post_fn=post_process)
	
	eta = M.ETA(ITER)
	for i in range(ITER):
		train_img, conf_map, geo_map = train_loader.get_next_batch()
		train_img = np.float32(train_img)
		conf_map = np.float32(conf_map)
		geo_map = np.float32(geo_map)
		losses, loss_total, tape = network.lossFunc(train_img, conf_map, geo_map, net)
		network.applyGrad(loss_total, net, optim, tape)

		if i%10==0:
			print('ITER:%d\tLS_conf:%.4f\tLS_xy:%.4f\tLS_wh:%.4f\tETA:%s'%(i, losses[0].numpy(), losses[1].numpy(), losses[2].numpy(), eta.get_ETA(i)))

		if i%SAVE_INTERVAL==0 and i>0:
			saver.save('./model/%d.ckpt'%i)

# if __name__=='__main__':
# 	net = network.network()
# 	optim = tf.train.AdamOptimizer(0.0001)
# 	saver = M.Saver(net, optim)
# 	saver.restore('./model/')
# 	x = np.ones([1,512,512,3],dtype=np.float32)
# 	y = net(x)
# 	print(y[0,:,:,0].numpy().max())
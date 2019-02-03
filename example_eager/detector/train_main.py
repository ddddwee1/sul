import network 
import modeleag as M 
import numpy as np 
import tensorflow as tf 

from data_reader import *

BSIZE = 16
ITER = 100001
SAVE_INTERVAL = 2000

def load_dataset():
	result = load_data()
	return result

def processed_bundle(bundle):
	img, annot = bundle
	img, annot = process_image(img, annot)
	img, conf, xywh = annot_to_grid(img, annot, 16)
	return bundle

def post_process(bundle):
	res = list(zip(*bundle))
	res = [np.float32(i) for i in res]
	return res 
	
if __name__=='__main__':
	if not os.path.exists('./result_img/'):
		os.mkdir('./result_img/')
	net = network.network()
	optim = tf.train.AdamOptimizer(0.0001)
	saver = M.Saver(net, optim)
	saver.restore('./model/')

	train_data = load_dataset()

	train_loader = M.DataReader(data=train_data, fn=processed_bundle, batch_size=BSIZE, \
		shuffle=True, random_sample=True, post_fn=post_process)
	
	eta = M.ETA(ITER)
	for i in range(ITER):
		train_img, conf_map, geo_map = train_loader.get_next_batch()
		losses, loss_total, tape, conf, geo = network.lossFunc(train_img, conf_map, geo_map, net)
		network.applyGrad(loss_total, net, optim, tape)

		if i%10==0:
			print('ITER:%d\tLS_conf:%.4f\tLS_xy:%.4f\tLS_wh:%.4f\tETA:%s'%(i, losses[0].numpy(), losses[1].numpy(), losses[2].numpy(), eta.get_ETA(i)))
			pred_annot = grid_to_annot(conf.numpy(), geo.numpy())
			true_annot = grid_to_annot(conf_map[0], geo_map[0])
			img = draw(train_img[0], true_annot, color=(0,255,0))
			img = draw(img, pred_annot, color=(0,0,255))
			cv2.imwrite('./result_img/%d.jpg'%i, img)

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
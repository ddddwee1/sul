import layers2 as L
# L.set_gpu('1')
import modeleag as M 
import tensorflow as tf 
import numpy as np 

import network_cls 
import network_rpn
import datareader

import cv2 
import random 

class Module(M.Model):
	def initialize(self):
		self.rpn_net = network_rpn.network()
		self.cls_net = network_cls.network()

	def forward(self, img, grid=16):
		# fix batch size at 1
		assert img.get_shape().as_list()[0]==1,'Batchsize more than 1 is not implemented yet'
		x = self.rpn_net(img)
		maps = x
		x = x[0]
		conf, xy, wh = x[:,:,0:1], x[:,:,1:3], x[:,:,3:]
		wh = tf.exp(wh) / 2.

		shape = xy.get_shape().as_list()
		
		patches = []

		coords = []

		for i in range(shape[0]):
			for j in range(shape[1]):
				# if conf[i,j,0].numpy()>0:
				if True:
					scale =  wh[i,j,1] *2 / 32.
					corner_x = j*grid + grid/2 - xy[i,j,0] - wh[i,j,0]
					corner_y = i*grid + grid/2 - xy[i,j,1] - wh[i,j,1]
					H = [scale, 0, corner_x, 0, scale, corner_y, 0, 0]
					out_shape = [32, int( wh.numpy()[i,j,0] * 2 / scale.numpy())]
					out = M.image_transform(img[0], H, out_shape, 'BILINEAR')
					out = tf.image.resize_images(out, [32, 100])
					patches.append(out)

					corner_x1 = corner_x + wh[i,j,0]*2
					corner_y1 = corner_y + wh[i,j,1]*2
					coords.append([corner_x.numpy(), corner_y.numpy(), corner_x1.numpy(), corner_y1.numpy()])

		cls_result = self.cls_net(patches)

		return maps, cls_result, coords
		# return patches

	def get_patches(self, img, grid=16):
		img = tf.convert_to_tensor(img)
		x = self.rpn_net(img)
		x = x[0]
		conf, xy, wh = x[:,:,0:1], x[:,:,1:3], x[:,:,3:]
		wh = tf.exp(wh) / 2.

		shape = xy.get_shape().as_list()
		
		patches = []

		for i in range(shape[0]):
			for j in range(shape[1]):
				# if conf[i,j,0].numpy()>0:
				if True:
					scale =  wh[i,j,1] *2 / 32.
					corner_x = j*grid + grid/2 - xy[i,j,0] - wh[i,j,0]
					corner_y = i*grid + grid/2 - xy[i,j,1] - wh[i,j,1]
					H = [scale, 0, corner_x, 0, scale, corner_y, 0, 0]
					out_shape = [32, int( wh.numpy()[i,j,0] * 2 / scale.numpy())]
					out = M.image_transform(img[0], H, out_shape, 'BILINEAR')
					out = tf.image.resize_images(out, [32, 100])
					patches.append(out)
		return patches

def computeIOU(box1_p, box2):
	x1, y1, x2, y2 = box1_p
	x3, y3,_, _, x4, y4,_,_,_ = box2 

	dx = max(0, min(x2,x4) - max(x1,x3))
	dy = max(0, min(y2,y4) - max(y1,y3))
	inter = dx*dy 
	union = (x4 - x3) * (y4 - y3) + (x2 - x1) * (y2 - y1) - inter
	iou = inter / (union + 1)
	return iou 

def post_process(bundle):
	res = list(zip(*bundle))
	res = [np.float32(i) for i in res]
	return res 

def get_ious(gt_boxes, pred_boxes):
	res = []
	for p in pred_boxes:
		ious = []
		for g in gt_boxes:
			iou = computeIOU(p, g)
			ious.append(iou)
		res.append([np.array(ious).max()])
	res = np.float32(res)
	return res 

def sup_loss(img, network, conf_map, geo_map, gt_boxes):
	with tf.GradientTape() as tape:
		maps, cls_result, coords = network(img)
		conf_pred = maps[:,:,:,0:1]
		geo_pred = maps[:,:,:,1:]
		
		# print(conf_map.shape)
		weight = tf.stop_gradient(tf.abs(tf.sigmoid(conf_pred) - conf_map))
		conf_loss = tf.reduce_sum( tf.nn.sigmoid_cross_entropy_with_logits(logits=conf_pred, labels=conf_map) * weight ) / (tf.reduce_sum(weight) + 1e-6)
		geo_loss = tf.reduce_sum( tf.square(geo_map - geo_pred) * conf_map ) / (tf.reduce_sum(conf_map) + 1e-6)

		ious = get_ious(gt_boxes, coords)
		ious2 = ious.copy() # I'm lazy

		# binary loss
		ious[ious<0.5] = 0
		ious[ious>0] = 1
		weight = tf.stop_gradient(tf.abs(tf.sigmoid(cls_result) - ious))
		cls_loss = tf.reduce_sum( tf.nn.sigmoid_cross_entropy_with_logits(logits=cls_result, labels=ious) * weight) / (tf.reduce_sum(weight) + 1e-6) # balance weight
		# # continuous loss (ce)
		# cls_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=cls_result, labels=ious) )
		# # continuous loss (mse)
		# cls_loss = tf.reduce_mean( tf.square(tf.sigmoid(cls_result) - ious) )

		# RPN traceback
		# eliminate non-overlapped samples. Large Degree of Freedom will cause non-convergence
		ious2[ious2<0.1] = 0.
		ious2[ious2>0.] = 1.
		cls_traceback = tf.reduce_sum(tf.square(tf.sigmoid(cls_result) - tf.ones_like(cls_result)) * ious2 ) / (ious2.sum() + 1e-6)

	return [conf_loss, geo_loss, cls_loss, cls_traceback], tape

def parse_conf_map(conf_map):
	confs = []
	shape = conf_map.get_shape().as_list()
	for i in range(shape[0]):
		for j in range(shape[1]):
			confs.append(conf_map[i,j,0])
	return confs 

def unsup_loss(img, network):
	with tf.GradientTape() as tape:
		maps, cls_result, coords = network(img)

		confs = maps[0,:,:,0:1]
		confs = parse_conf_map(confs)
		
		# confidence consistency
		conf_consist = [tf.square(tf.sigmoid(confs[i]) - tf.stop_gradient(tf.sigmoid(cls_result[i]))) for i in range(len(confs))]
		conf_consist = sum(conf_consist) / len(conf_consist)

		# RPN traceback
		scores = [i.numpy() for i in cls_result]
		scores = np.float32(scores)
		scores[scores<0.3] = 0
		scores[scores>0] = 1
		cls_traceback = tf.reduce_sum(tf.square(tf.sigmoid(cls_result) - tf.ones_like(cls_result)) * scores ) / (scores.sum() + 1e-6)
	return [conf_consist, cls_traceback], tape

def applyGrad_sup(mod, losses, optim, tape):
	variables = [mod.rpn_net.variables, mod.rpn_net.variables, mod.cls_net.variables, mod.rpn_net.variables]
	grads = tape.gradient(losses, variables)
	for g,v in zip(grads, variables):
		optim.apply_gradients(zip(g,v))

def applyGrad_unsup(mod, losses, optim, tape):
	variables = [mod.rpn_net.variables, mod.rpn_net.variables]
	grads = tape.gradient(losses, variables)
	for g,v in zip(grads, variables):
		optim.apply_gradients(zip(g,v))

if __name__=='__main__':
	mod = Module()
	optim = tf.train.AdamOptimizer(0.00003)

	# saver = M.Saver(mod.rpn_net)
	# saver.restore('./model_rpn/')
	# del saver 

	saver = M.Saver(mod, optim)
	saver.restore('./model/')

	loader = M.DataReader(data=datareader.data_reader(img_fn='./task1/train/img/'), fn = datareader.processe_bundle,\
		batch_size=1, shuffle=True, random_sample=True, post_fn=post_process, processes=1)

	unlabelled_loader = M.DataReader(data=datareader.data_reader(img_fn='./unlabel/task1/train/img/'), fn = datareader.processe_bundle,\
		batch_size=1, shuffle=True, random_sample=True, post_fn=post_process, processes=1)

	# start supervised training
	# MAXITER = 2001
	# for i in range(MAXITER):
	# 	train_img, conf_map, geo_map = loader.get_next_batch()
	# 	gt_boxes = datareader.map_to_box(conf_map[0], geo_map[0], 0.0)
	# 	if len(gt_boxes)==0:
	# 		continue
	# 	losses , tape = sup_loss(train_img, mod, conf_map, geo_map, gt_boxes)
	# 	applyGrad_sup(mod, losses, optim, tape)
	# 	print('ITER:%d\tConfLoss:%.4f\tGeoLoss:%.4f\tClsLoss:%.4f\tTBLoss:%.4f'%(i,losses[0].numpy(),losses[1].numpy(),losses[2].numpy(),losses[3].numpy()))
	# 	if i%2000==0 and i>0:
	# 		saver.save('./model/model.ckpt')
		
	# start unsupervised training
	MAXITER = 10001
	for i in range(MAXITER):
		train_img, conf_map, geo_map = loader.get_next_batch()
		gt_boxes = datareader.map_to_box(conf_map[0], geo_map[0], 0.0)
		if len(gt_boxes)==0:
			continue
		losses , tape = sup_loss(train_img, mod, conf_map, geo_map, gt_boxes)
		applyGrad_sup(mod, losses, optim, tape)
		train_img, _, _ = loader.get_next_batch()
		losses_unsup, tape = unsup_loss(train_img, mod)
		applyGrad_unsup(mod, losses_unsup, optim, tape)
		print( 'ITER:%d\tConfLoss:%.4f\tGeoLoss:%.4f\tClsLoss:%.4f\tTBLoss:%.4f\tConstLoss:%.4f\tTBLoss_unsup:%.4f'%(i,losses[0].numpy(),losses[1].numpy(),losses[2].numpy(),losses[3].numpy(),losses_unsup[0].numpy(), losses_unsup[1].numpy()) )
		
		if i%100==0:
			maps = mod.rpn_net(train_img)
			text_polys = datareader.map_to_box(maps[0,:,:,0:1], maps[0,:,:,1:], 0.3)
			res_img = datareader.plot(train_img[0], text_polys, (255,255,0))
			cv2.imwrite('./res/%d.jpg'%i, res_img)
		if i%2000==0 and i>0:
			saver.save('./model/model.ckpt')

'''
TO-DO:
1. EMA model
2. Psudo label pool
'''

import numpy as np 
import netpart

import model as M
import tensorflow as tf 
import cv2 
import time
import os 
import data_reader

if __name__=='__main__':
	print('data_reader')
	reader = M.data_reader(data=data_reader.get_data(), fn=data_reader.process,
							 batch_size=4, shuffle=True, post_fn=data_reader.post_process, processes=3)

	def sec2hms(sec):
		hm = sec//60
		s = sec%60
		h = hm // 60
		m = hm % 60
		return h,m,s

	inpholder, labholder, netout, loss, train_step = netpart.get_net()

	start_time = time.time()
	MAX_ITER = 200000
	with tf.Session() as sess:
		saver = tf.train.Saver()
		M.loadSess('./model/',sess,init=True)
		for i in range(MAX_ITER):
			img_batch, hmap_batch = reader.get_next_batch()
			ls, out, _ = sess.run([loss, netout, train_step], feed_dict={inpholder: img_batch, labholder: hmap_batch})
			if i%10==0:
				t2 = time.time()
				remain_time = float(MAX_ITER - i) / float(i+1) * (t2 - start_time)
				h,m,s = sec2hms(remain_time)
				print('Iter:\t%d\tLoss:\t%.6f\tETA:%d:%d:%d'%(i,ls,h,m,s))
			if i%10==0:
				img = img_batch[0]
				gt = hmap_batch[0]
				out = out[0]
				cv2.imshow('img',img)
				cv2.imshow('gt',gt)
				cv2.imshow('out',out)
				cv2.waitKey(1)
			if i%5000==0 and i>0:
				saver.save(sess,'./model/MSRPN_%d.ckpt'%i)


# inpholder, labholder, netout, loss, train_step = netpart.get_net()
# netout = tf.sigmoid(netout)
# img = cv2.imread('1.jpg')
# canvas = np.zeros([2666,2666,3], dtype=np.uint8)
# canvas[333:-333, :] = img
# img = cv2.resize(canvas, (512, 512))
# with tf.Session() as sess:
# 	M.loadSess('./model/',sess=sess)
# 	out = sess.run(netout, feed_dict={inpholder: [img]})
# 	cv2.imshow('img',img)
# 	cv2.imshow('out',out[0])
# 	cv2.waitKey(0)
# 	out = out[0]
# 	out = out / out.max() * 255
# 	out = np.uint8(out)
# 	cv2.imwrite('1_o.jpg',out)


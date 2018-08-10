import numpy as np 
import cv2 
import progressbar
import random

def cvt_age(age):
	if age<20:
		lb = 0
	elif 20<age<25:
		lb = 1
	elif 25<=age<30:
		lb = 2
	elif 30<=age<40:
		lb = 3
	elif 40<=age<50:
		lb = 4
	elif 50<=age<60:
		lb = 5
	else:
		lb = 6
	return lb

class data_reader():
	def __init__(self,fname='train_list.txt'):
		bar = progressbar.ProgressBar(max_value=201000)

		f = open(fname)
		cnt = 0
		id_num = 0
		d_IMG = {}
		d_ID = {}
		l_ID = []
		for i in f:
			cnt += 1
			bar.update(cnt)
			i = i.strip().split('\t')[0]
			img = cv2.imread(i)
			i = i.replace('\\','/').split('/')[-1]
			age = int(i.split('.')[0].split('_')[1][-2:])
			ID = int(i.split('_')[0])
			age = cvt_age(age)

			img_pack = [img,age]
			if not ID in d_IMG:
				d_IMG[ID] = []
				d_ID[ID] = id_num
				id_num += 1
				l_ID.append(ID)
			d_IMG[ID].append(img_pack)
		f.close()

		self.d_IMG = d_IMG
		self.d_ID = d_ID
		self.l_ID = l_ID 
		self.max_id = id_num
		self.eye7 = np.eye(7)
		self.eyeid = np.eye(id_num)

		print('\nData reading finished.')
		print('ID:',id_num)
		print('Data length:',cnt)
		self.age_class = 7

	def get_train_batch(self,BSIZE):
		batch_id = random.sample(self.l_ID,BSIZE)
		batch_img = [random.sample(self.d_IMG[k],1)[0] for k in batch_id]
		batch_target = [random.sample(self.d_IMG[k],1)[0] for k in batch_id]
		batch_age = [i[1] for i in batch_target]
		batch_target = [i[0] for i in batch_target]
		batch_img = [i[0] for i in batch_img]
		batch_id = [self.d_ID[k] for k in batch_id]

		batch_img = np.float32(batch_img)
		batch_target = np.float32(batch_target)
		batch_uniform = np.random.uniform(-10.,10.,size=(BSIZE,2,2,512))
		batch_age = np.int32(batch_age)
		batch_age = self.eye7[batch_age]
		batch_id = np.int32(batch_id)
		batch_id = self.eyeid[batch_id]
		return batch_img, batch_target, batch_uniform, batch_age, batch_id

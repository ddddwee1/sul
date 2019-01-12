# class for data reader 
# include online data augmentation steps

import numpy as np 
import cv2 
import random

class data_reader():
	def __init__(self, validation_ratio):
		data = self.load_dataset() # data: list of [image, label]

		random.shuffle(data)
		val_index = int(len(data)*validation_ratio)
		self.training_data = data[val_index:]
		self.validation_data = data[:val_index]

		self.training_position = 0
		self.validation_position = 0

	def random_scale(self, image, label):
		image_processed = image.copy()
		label_processed = label.copy()
		# put your processing code here
		return image_processed, label_processed

	def random_crop(self, image, label):
		image_processed = image.copy()
		label_processed = label.copy()
		# put your processing code here
		return image_processed, label_processed

	def interpret_label(self, label):
		# input: label
		# output: mask (128,128), geo (128,128,4), rot (128,128,1), quad (128,128,8)

		## put your processing code here

		return mask, geo, rot, quad

	def process_batch(self. batch):
		result = []

		for img,label in batch:
			image_processed, label_processed = self.random_scale(img, label)
			image_processed, label_processed = self.random_crop(image_processed, label_processed)
			mask, geo, rot, quad = self.interpret_label(label_processed)
			result.append([image_processed, mask, geo, rot, quad])

		image_batch = [i[0] for i in result]
		mask_batch = [i[1] for i in result]
		geo_batch = [i[2] for i in result]
		rot_batch = [i[3] for i in result]
		quad_batch = [i[4] for i in result]

		return image_batch, mask_batch, geo_batch, rot_batch, quad_batch 

	def get_training_batch(self, batchsize):
		if self.training_position+batchsize >= len(self.training_data):
			self.training_position = 0
			random.shuffle(self.training_data)
		
		batch = self.training_data[self.training_position: self.training_position+batchsize]
		self.training_position += batchsize
		return self.process_batch(batch)

	def get_validation_batch(self, batchsize):
		if self.validation_position+batchsize >= len(self.validation_data):
			self.validation_position = 0
			random.shuffle(self.validation_position)

		batch = self.validation_data[self.validation_position: self.validation_position+batchsize]
		self.validation_position += batchsize
		return self.process_batch(batch)

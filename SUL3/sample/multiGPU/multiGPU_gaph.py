import tensorflow as tf 
import layers3 as L 
import numpy as np 
import model3 as M 
import os 

class cnnModel(M.Model):
	def initialize(self):
		self.c1 = M.ConvLayer2D(5, 32, stride=2, activation=M.PARAM_RELU)
		self.c2 = M.ConvLayer2D(5, 64, stride=2, activation=M.PARAM_RELU)
		self.c3 = M.ConvLayer2D(5,128, stride=2, activation=M.PARAM_RELU)
		self.fc = M.Dense(10)

	def forward(self, x):
		x = self.c1(x)
		x = self.c2(x)
		x = self.c3(x)
		x = M.flatten(x)
		x = self.fc(x)
		x = tf.nn.softmax(x, axis=-1)
		return x 


mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images[..., None]
test_images = test_images[..., None]

train_images = train_images / 255. 
test_images = test_images / 255.

strategy = tf.distribute.MirroredStrategy()
print('Number of devices:', strategy.num_replicas_in_sync)

BUFFER_SIZE = len(train_images)

BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

EPOCHS = 10
train_steps_per_epoch = len(train_images) // BATCH_SIZE
test_steps_per_epoch = len(test_images) // BATCH_SIZE

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

with strategy.scope():
	train_iterator = strategy.experimental_make_numpy_iterator(
		(train_images, train_labels), BATCH_SIZE, shuffle=BUFFER_SIZE)

	test_iterator = strategy.experimental_make_numpy_iterator(
		(test_images, test_labels), BATCH_SIZE, shuffle=None)

	loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
	train_loss = tf.keras.metrics.Mean(name='train_loss')
	test_loss = tf.keras.metrics.Mean(name='test_loss')

	train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
	test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
	model = cnnModel()
	optimizer = tf.keras.optimizers.Adam(0.001)
	checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

	def train_step(inputs):
		images, labels = inputs

		with tf.GradientTape() as tape:
			predictions = model(images)
			loss = loss_object(labels, predictions)

		gradients = tape.gradient(loss, model.trainable_variables)
		optimizer.apply_gradients(zip(gradients, model.trainable_variables))

		train_loss(loss)
		train_accuracy(labels, predictions)

	# Test step
	def test_step(inputs):
		images, labels = inputs

		predictions = model(images)
		t_loss = loss_object(labels, predictions)

		test_loss(t_loss)
		test_accuracy(labels, predictions)

	@tf.function
	def distributed_train():
		return strategy.experimental_run(train_step, train_iterator)
	
	@tf.function
	def distributed_test():
		return strategy.experimental_run(test_step, test_iterator)
		
	for epoch in range(EPOCHS):
		# Note: This code is expected to change in the near future.
		
		# TRAIN LOOP
		# Initialize the iterator
		train_iterator.initialize()
		for _ in range(train_steps_per_epoch):
			distributed_train()

		# TEST LOOP
		test_iterator.initialize()
		for _ in range(test_steps_per_epoch):
			distributed_test()
		
		if epoch % 2 == 0:
			checkpoint.save(checkpoint_prefix)

		template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
								"Test Accuracy: {}")
		print (template.format(epoch+1, train_loss.result(), 
													 train_accuracy.result()*100, test_loss.result(), 
													 test_accuracy.result()*100))
		
		train_loss.reset_states()
		test_loss.reset_states()
		train_accuracy.reset_states()
		test_accuracy.reset_states()

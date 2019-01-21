import tensorflow as tf 
import numpy as np 
import model as M 
import pandas as pd 
import matplotlib.pyplot as plt 

def read_data():
	file = pd.read_csv('lin_data.csv')
	x = file['x']
	y = file['y']
	return x,y

def plot_data(x,y):
	plt.plot(x,y,'x')
	plt.show()

def build_model(input_data):
	mod = M.Model(input_data,[None,1])
	mod.fcLayer(1)
	return mod.get_current_layer()

def build_graph():
	input_placeholder = tf.placeholder(tf.float32,[None,1])
	label_placeholder = tf.placeholder(tf.float32,[None,1])
	output = build_model(input_placeholder)
	loss = tf.reduce_mean(tf.square(output - label_placeholder))
	train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
	return input_placeholder, label_placeholder, output, loss, train_step

MAX_ITER = 10000
input_placeholder, label_placeholder, output, loss, train_step = build_graph()
x,y = read_data()

x_reform = [[item] for item in x]
y_reform = [[item] for item in y]

import time
with tf.Session() as sess:
	a = time.time()
	M.loadSess('./model/',sess,init=True)
	
	for i in range(MAX_ITER):
		feed_d = {input_placeholder:x_reform, label_placeholder:y_reform}
		ls, _ = sess.run([loss,train_step],feed_dict = feed_d)
		if i%100==0:
			print('Loss at Iter',i,':',ls)
	b = time.time()
	print(b-a)

	x_test = [[num*0.001] for num in range(20000)]
	y_test = sess.run(output,feed_dict={input_placeholder:x_test})
	x_test = [item[0] for item in x_test]
	y_test = [item[0] for item in y_test]
	plt.plot(x_test,y_test)
	plot_data(x,y)

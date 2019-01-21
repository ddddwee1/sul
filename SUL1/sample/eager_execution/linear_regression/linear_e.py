import tensorflow as tf 
import numpy as np 
import model as M 
import pandas as pd 
import matplotlib.pyplot as plt 
import time

tf.enable_eager_execution()

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

def build_loss(x,y):
	with tf.GradientTape() as tape:
		output = build_model(x)
		loss = tf.reduce_mean(tf.square(output - y))
	return output, loss, tape

MAX_ITER = 10000
x,y = read_data()

x_reform = np.float32([[item] for item in x])
y_reform = np.float32([[item] for item in y])

optim = tf.train.GradientDescentOptimizer(0.001)
var = M.VAR_LIST

a = time.time()
with tf.device("/gpu:0"):
	for i in range(MAX_ITER):
		out, ls, tape = build_loss(x_reform, y_reform)
		# print(len(var))
		# input()
		grad = tape.gradient(ls,var)
		optim.apply_gradients(zip(grad,var))
		if i%100==0:
			print('Loss at Iter %d: %.4f'%(i,ls))
b = time.time()
print(b-a)

x_test = [[num*0.001] for num in range(20000)]
y_test = build_model(x_test)
x_test = [item[0] for item in x_test]
y_test = [item[0] for item in y_test]
plt.plot(x_test,y_test)
plot_data(x,y)

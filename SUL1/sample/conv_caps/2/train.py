from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import net

net = net.network()

BSIZE = 128
MAX_ITER = 20000

for i in range(MAX_ITER):
	x_train, y_train = mnist.train.next_batch(BSIZE)
	x_train = x_train.reshape([-1,28,28,1])
	ls,ac = net.train(x_train,y_train)
	print('ITER:\t%d\tLoss:\t%.4f\tAcc:\t%.4f'%(i,ls,ac))

import caffe 
import numpy as np 

net = caffe.Net('LSTM_deploy1.prototxt', 'LSTM_PENN.caffemodel', caffe.TEST)

print(net.blobs['data1'].data.shape)

net.blobs['data1'].data[:] = np.ones([1,3,368, 368]).astype(np.float32)
net.blobs['data2'].data[:] = np.ones([1,3,368, 368]).astype(np.float32)
net.blobs['center_map'].data[:] = np.ones([1,1,368, 368]).astype(np.float32)

net.forward()

def tstlayer(name):
	a = net.blobs[name].data 
	print(a)
	# print(a)
	print(a.shape)

tstlayer('Mconv5_stage2')

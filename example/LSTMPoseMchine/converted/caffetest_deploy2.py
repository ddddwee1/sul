import caffe 
import numpy as np 

net = caffe.Net('LSTM_deploy2.prototxt', 'LSTM_PENN.caffemodel', caffe.TEST)

print(net.blobs['data'].data.shape)

x = np.ones([1,3,368, 368]).astype(np.float32)
x[:,:,-1] = 0
net.blobs['data'].data[:] = x
net.blobs['heatmap'].data[:] = np.ones([1,15,46, 46]).astype(np.float32)
net.blobs['center_map'].data[:] = np.ones([1,1,368, 368]).astype(np.float32)
net.blobs['h_t_1'].data[:] = np.ones([1,48,46, 46]).astype(np.float32)
net.blobs['cell_t_1'].data[:] = np.ones([1,48,46, 46]).astype(np.float32)

net.forward()

def tstlayer(name):
	a = net.blobs[name].data 
	print(a)
	# print(a)
	print(a.shape)

tstlayer('Mres5_stage3')

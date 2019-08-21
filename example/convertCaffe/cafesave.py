import caffe 
import numpy as np 
import pickle 

TEST_LAYER_NAME = 'fc1452'
# TEST_LAYER_NAME = None

net = caffe.Net('abc.prototxt', caffe.TEST)
params = pickle.load(open('weights.pkl', 'rb'))

for key in params:
	if 'kernel' in params[key]:
		net.params[key][0].data[:] = np.transpose(params[key]['kernel'], [3,2,1,0])
	elif 'dwkernel' in params[key]:
		net.params[key][0].data[:] = np.transpose(params[key]['dwkernel'], [2,3,1,0])
	elif 'fckernel' in params[key]:
		net.params[key][0].data[:] = np.transpose(params[key]['fckernel'], [1,0])
	elif 'mean' in params[key]:
		net.params[key][0].data[:] = params[key]['mean']
		net.params[key][1].data[:] = params[key]['var']
		if 'scale' in params[key]:
			net.params[key][2].data[:] = params[key]['scale']
	elif 'scale' in params[key]:
		net.params[key][0].data[:] = params[key]['scale']
	if 'bias' in params[key]:
		net.params[key][1].data[:] = params[key]['bias']
	if 'gamma' in params[key]: # used for prelu, not sure if other layers use this too
		net.params[key][0].data[:] = params[key]['gamma']

print(net.blobs['inputholder'].data.shape)

net.blobs['inputholder'].data[:] = np.ones([1,3,128, 128]).astype(np.float32)

net.forward()

def tstlayer(name):
	if name is None:
		return 
	a = net.blobs[name].data 
	# a = np.transpose(a, [0,2,3,1])
	print(a)
	# print(a)
	print(a.shape)

tstlayer(TEST_LAYER_NAME)
net.save('abc.caffemodel')

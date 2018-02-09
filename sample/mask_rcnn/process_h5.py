import h5py 
import numpy as np 
import scipy.io as sio
import pickle

layername = ['conv1','bn_conv1']

def conv_block(stage,block,shortcut=True):
	conv_name_base = 'res'+str(stage)+str(block)+'_branch'
	bn_name_base = 'bn'+str(stage)+str(block)+'_branch'
	layername.append(conv_name_base+'2a')
	layername.append(bn_name_base+'2a')
	layername.append(conv_name_base + '2b')
	layername.append(bn_name_base + '2b')
	layername.append(conv_name_base + '2c')
	layername.append(bn_name_base + '2c')
	if shortcut:
		layername.append(conv_name_base + '1')
		layername.append(bn_name_base + '1')

def identity_block(stage,block):
	conv_block(stage,block,False)

conv_block(2,'a')
identity_block(2,'b')
identity_block(2,'c')
conv_block(3,'a')
identity_block(3,'b')
identity_block(3,'c')
identity_block(3,'d')
conv_block(4,'a')
for i in range(22):
	identity_block(4,chr(98+i))
conv_block(5,'a')
identity_block(5,'b')
identity_block(5,'c')

conv_list = ['kernel:0','bias:0']
bn_list = ['gamma:0','beta:0','moving_mean:0','moving_variance:0']

# layerdict = {'conv1':['kernel:0','bias:0'],
# 			'bn_conv1':['gamma:0','beta:0','moving_mean:0','moving_variance:0']}

fpnlayers = ['fpn_c5p5','fpn_c4p4','fpn_c3p3','fpn_c2p2','fpn_p2','fpn_p3','fpn_p4','fpn_p5']
layername += fpnlayers
layerdict = {}
for i in layername:
	if 'bn' in i:
		layerdict[i] = bn_list
	else:
		layerdict[i] = conv_list

print(layerdict)

f = h5py.File('a.h5')
for i in f.keys():
	print(i)

# f = f['bn_conv1']['bn_conv1']
# print('---')
# for i in f.keys():
# 	print(i)

cnt = 0
d = {}
for layer in layername:
	for item in layerdict[layer]:
		aa = np.float32(f[layer][layer][item])
		d[str(cnt)] = aa
		print(aa.shape)
		# input()
		cnt += 1
# sio.savemat('layer1',d)
save_file = 'buffer_weights.pickle'
with open(save_file,'wb') as f:
	pickle.dump(d,f)
print('Total length:',cnt)

# f2 = f['beta:0']
# f2 = np.array(f2)
# print(f2)


# f = f['conv1']['conv1']['kernel:0']
# f = np.array(f)
# print(f.shape)

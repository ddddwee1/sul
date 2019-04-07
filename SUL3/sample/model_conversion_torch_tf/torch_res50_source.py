import torch 
import torchvision 
import pickle

def process_data(layer_name,layer_param, layer_buf):
	print(layer_name)
	for p in (layer_param + layer_buf):
		name = p[0]
		value = p[1]
		if isinstance(value, torch.Tensor):
			data = value.detach().cpu().numpy()
		else:
			data = value.detach().cpu().data.numpy()
		
		fout = open('./params/%s_layersplit_%s.pkl'%(layer_name,name), 'wb')
		pickle.dump(data, fout)
		fout.close()
		print(data.shape)

def process_module(parent_name, module):
	module_name = module[0]
	module_obj = module[1]
	children = list(module_obj.named_children())
	if len(children)==0:
		process_data(parent_name+'_modulesplit_'+module_name, list(module_obj.named_parameters()), list(module_obj._buffers.items()))
	else:
		for child in children:
			process_module(parent_name+'_modulesplit_'+module_name, child)

model = torchvision.models.resnet50(pretrained=True)
model.eval()

######## for testing #######
# submodules = list(model.children())

# x = torch.ones((1,3,224,224), dtype=torch.float)
# for i in range(8):
# 	x = submodules[i](x)

# print(x)
# print(x.shape)

######## for extraction #######
if not os.path.exists('./params/'):
	os.mkdir('./params/')
process_module('',['',model])

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init 
from torch.nn.parameter import Parameter
import math 

class Model(nn.Module):
	def __init__(self, *args, **kwargs):
		super(Model, self).__init__()
		self.is_built = False
		self.initialize(*args, **kwargs)

	def initialize(self, *args, **kwargs):
		pass 

	def build(self, *inputs):
		pass 

	def __call__(self, *input, **kwargs):
		if not self.is_built:
			self.build(*input)
		for hook in self._forward_pre_hooks.values():
			result = hook(self, input)
			if result is not None:
				if not isinstance(result, tuple):
					result = (result,)
				input = result
		if torch._C._get_tracing_state():
			result = self._slow_forward(*input, **kwargs)
		else:
			result = self.forward(*input, **kwargs)
		for hook in self._forward_hooks.values():
			hook_result = hook(self, input, result)
			if hook_result is not None:
				result = hook_result
		if len(self._backward_hooks) > 0:
			var = result
			while not isinstance(var, torch.Tensor):
				if isinstance(var, dict):
					var = next((v for v in var.values() if isinstance(v, torch.Tensor)))
				else:
					var = var[0]
			grad_fn = var.grad_fn
			if grad_fn is not None:
				for hook in self._backward_hooks.values():
					wrapper = functools.partial(hook, self)
					functools.update_wrapper(wrapper, hook)
					grad_fn.register_hook(wrapper)
		self.is_built = True
		return result

class conv2D(Model):
	def initialize(self, size, outchn, stride=1, pad='SAME_LEFT', dilation_rate=1, usebias=True, gropus=1):
		self.size = size
		self.outchn = outchn
		self.stride = stride
		self.usebias = usebias
		self.gropus = gropus
		self.dilation_rate = dilation_rate
		assert (pad in ['VALID','SAME_LEFT'])
		self.pad = pad 

	def _parse_args(self, input_shape):
		inchannel = input_shape[1]
		# parse args
		if isinstance(self.size,list):
			# self.size = [self.size[0],self.size[1],inchannel,self.outchn]
			self.pad = (self.size[0]//2, self.size[1]//2)
			self.size = [self.outchn, inchannel // self.gropus, self.size[0], self.size[1]]
		else:
			self.pad = self.size//2
			self.size = [self.outchn, inchannel // self.gropus, self.size, self.size]

	def build(self, *inputs):
		# print('building...')
		inp = inputs[0]
		self._parse_args(inp.shape)
		self.weight = Parameter(torch.Tensor(*self.size))
		if self.usebias:
			self.bias = Parameter(torch.Tensor(self.outchn))
		else:
			self.register_parameter('bias', None)
		self.reset_params()
		
	def reset_params(self):
		init.kaiming_uniform_(self.weight, a=math.sqrt(5))
		if self.bias is not None:
			fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
			bound = 1 / math.sqrt(fan_in)
			init.uniform_(self.bias, -bound, bound)

	def forward(self, x):
		return F.conv2d(x, self.weight, self.bias, self.stride, self.pad, self.dilation_rate, self.gropus)

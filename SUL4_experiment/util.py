import time 
import tensorflow as tf 
import json 

################
# ETA class. I want to see the ETA. It's too boring to wait here.
class ETA():
	"""
	Class for estimating the ETA. 
	"""
	def __init__(self,max_value):
		"""
		:param max_value: Max iteration
		"""
		self.start_time = time.time()
		self.max_value = max_value
		self.current = 0

	def start(self):
		"""
		Reset the start time. In case that time-consumable operations are taken between initialization and update.
		"""
		self.start_time = time.time()
		self.current = 0

	def sec2hms(self,sec):
		"""
		Second to hour:minute:second

		:type sec: int
		:param sec: Number of seconds to be converted.

		:return: A tuple containing hour, minute and second value.
		"""
		hm = sec//60
		s = sec%60
		h = hm//60
		m = hm%60
		return h,m,s

	def get_ETA(self,current,is_string=True):
		"""
		Get ETA based on current iteration.

		:type current: int
		:param current: Current number of iterations.

		:type is_string: bool
		:param is_string: Whether to return a string or tuple
		"""
		self.current = current
		time_div = time.time() - self.start_time
		time_remain = time_div * float(self.max_value - self.current) / float(self.current + 1)
		h,m,s = self.sec2hms(int(time_remain))
		if is_string:
			return '%d:%d:%d'%(h,m,s)
		else:
			return h,m,s

################
class EMAMeter():
	"""
	Exponential moving average meter.
	"""
	def __init__(self, alpha):
		"""
		:type alpha: float
		:param alpha: The exponential value (or known as decay value) for moving average.
		"""
		self.alpha = alpha
		self.value = None

	def update(self, value):
		"""
		:type value: float
		:param value: The observation value.

		:return: Current value of the EMA meter.
		"""
		if self.value is None:
			self.value = value
		else:
			self.value = self.value * self.alpha + value * (1-self.alpha)
		return self.value

################
class Summary():
	def __init__(self, fname, save_interval=100):
		self.n_iter = 0
		self.fname = fname
		self.save_interval = save_interval
		self.data = {}
	def push(self, category, value):
		value = float(value)
		if not category in self.data:
			self.data[category] = []
		self.data[category].append([self.n_iter, value])
	def step(self):
		self.n_iter += 1
		if self.n_iter%self.save_interval==0:
			with open(self.fname,'w') as f:
				json.dump(self.data, f)

######## LR Scheduler ########
class LRScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
	def __init__(self, fn):
		super(LRScheduler, self).__init__()
		self.schedule_fn = fn 

	def __call__(self, step):
		lr = self.schedule_fn(step)
		return lr 

	def get_config(self):
		return {'lr_fn':self.schedule_fn}

def zip_grad(grads, vars):
	assert len(grads)==len(vars)
	grads_1 = []
	vars_1 = []
	for i in range(len(grads)):
		if not grads[i] is None:
			grads_1.append(grads[i])
			vars_1.append(vars[i])
	assert len(grads_1)!=0
	return zip(grads_1, vars_1)

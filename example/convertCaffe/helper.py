class LayerName():
	def __init__(self, name):
		self.shape = None 
		self.name = name 
	def __call__(self):
		return self.name 

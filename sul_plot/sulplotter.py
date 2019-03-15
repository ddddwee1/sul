import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 

class Plotter3D():
	def __init__(self, usebuffer=False, elev=None, azim=None, axis='on'):
		fig = plt.figure()
		self.ax = fig.add_subplot(111, projection='3d')
		self.ax.view_init(elev=elev, azim=azim)
		self.ax.axis(axis)
		self.lines = []
		self.lines_buff = []
		self.line_pos = 0
		self.usebuffer = usebuffer
		self.fig = fig 

	def show(self, ion=True):
		if ion:
			plt.ion()
		plt.show()

	def clear(self):
		self.ax.clear()

	def plot(self, xs,ys,zs, lims=None, **kwargs):
		if lims is not None:
			self.ax.set_xlim(lims[0])
			self.ax.set_ylim(lims[1])
			self.ax.set_zlim(lims[2])

		# self.ax.plot(xs, ys, zs, **kwargs)

		if (len(self.lines)==0) or (not self.usebuffer):
			a = self.ax.plot(xs, ys, zs, **kwargs)
			self.lines_buff.append(a)
		else:
			line = self.lines[self.line_pos][0]
			line.set_data(xs,ys)
			line.set_3d_properties(zs)
			self.line_pos += 1
		
	def update(self, require_img=False):
		# slowest 
		# plt.pause(0.0001)
		# faster
		# self.fig.canvas.draw()
		# much faster
		try:
			self.ax.draw_artist(self.ax.patch)
			for line in self.lines:
				self.ax.draw_artist(line)
			self.fig.canvas.update()
		except:
			self.fig.canvas.draw()

		if require_img:
			# image = np.fromstring(self.fig.canvas.tostring_rgb(), dtype='uint8')
			s, (width, height) = self.fig.canvas.print_to_buffer()
			image = np.fromstring(s, np.uint8).reshape((height, width, 4))

		self.fig.canvas.flush_events()
		if len(self.lines)==0:
			self.lines = self.lines_buff
			self.lines_buff = []
		self.line_pos = 0

		if require_img:
			return image

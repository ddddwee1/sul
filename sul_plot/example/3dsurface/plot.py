import sulplotter as plotter 
import numpy as np 
from scipy.interpolate import griddata

X = [0.1, 0.2, 0.3, 0.4, 0.5]
Y = [16, 32, 64, 128, 256]

X,Y = np.meshgrid(X,Y)

coord = np.stack([X,Y], axis=-1).reshape([-1,2])
# print(coord.shape)

scr = [[60.1, 57.6, 55.4, 56.3, 58.5],
		[58.8, 54.9, 51.8, 53.1, 55.9],
		[56.1, 52.3, 48.0, 49.1, 51.2],
		[48.2, 45.7, 42.9, 44.2, 46.1],
		[49.1, 46.3, 44.1, 45.8, 48.0]]

# scr = [[47.2, 45.3, 43.1, 44.3, 46.9],
# 		[41.9, 41.1, 40.8, 41.6, 43.8],
# 		[40.8, 39.9, 37.3, 38.6, 40.1],
# 		[39.1, 38.6, 34.1, 36.6, 37.0],
# 		[39.3, 39.1, 34.6, 36.4, 36.8]]
scr = np.float32(scr).reshape([-1])

X_plt = np.arange(0.1, 0.5, 0.01)
Y_plt = np.arange(16, 256, 1)
X_plt,Y_plt = np.meshgrid(X_plt,Y_plt)
grid_z = griddata(coord, scr, (X_plt, Y_plt), method='cubic')
print(grid_z.shape)

from matplotlib import cm
plt = plotter.Surface3D(azim=45)

plt.plot(X_plt,Y_plt, grid_z,cmap=cm.coolwarm,linewidth=0, antialiased=True)
plt.set_label(['Threshold', 'Time', 'Error'])
plt.show(False)

import sulplotter

plotter = sulplotter.LossPlotterJson('log.json', keys=['acc','loss','loss_ttl'], scales=[1,1,100])
plotter.plot()
plotter.set_legend('upper right')
plotter.show()

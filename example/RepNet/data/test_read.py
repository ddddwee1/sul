import numpy as np 
import pickle 

data = pickle.load(open('points_flatten2.pkl', 'rb'))

print(len(data))

print(data[0][0])

print(data[0][1])

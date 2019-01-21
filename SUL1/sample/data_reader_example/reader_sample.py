import data_reader
import time
import tensorflow as tf 

def worker(num):
	time.sleep(0.5)
	print(num)
	return num 

if __name__=='__main__':
	data = list(range(100))
	bsize = 10

	reader = data_reader.data_reader(data, worker, bsize)

	for i in range(10):
		a = reader.get_next_batch()
		print(a)

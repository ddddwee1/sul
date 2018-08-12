import model as M 
import tensorflow as tf 
import numpy as np 

max_digit = 6

int2binary = {}
largest_num = pow(2,max_digit)
binary = np.unpackbits(np.array([range(largest_num)],dtype=np.uint8).T,axis=1)
for i in range(largest_num):
	int2binary[i] = binary[i]

def getInputVector():
	maxnum = pow(2,max_digit)
	a = np.random.randint(maxnum/2)
	b = np.random.randint(maxnum/2)
	aarr = int2binary[a]
	barr = int2binary[b]
	c = a+b
	carr = int2binary[c]
	x = list()
	y = list()
	for i in range(max_digit):
		x.append(np.array([aarr[-i-1],barr[-i-1]]))
		y.append(np.array([carr[-i-1]]))
	# y = [[-1] if k[0]==0 else [1] for k in y]
	return np.array(x),np.array(y)

re_mod = False
def mod(inp,reuse=False):
	global re_mod
	with tf.variable_scope('last',reuse=reuse):
		mod = M.Model(inp)
		mod.fcLayer(1)
		re_mod = True
	return mod.get_current_layer()

class LSTM():
	def __init__(self,dim,name):
		self.reuse = False
		self.name = name
		self.dim = dim

	def apply(self, inp, hidden=None, cell=None):
		bsize = tf.shape(inp)[0]
		zero_state = tf.constant(0., shape=[1,self.dim])
		zero_state = tf.tile(zero_state, [bsize, 1])
		if hidden is None:
			hidden = zero_state
		if cell is None:
			cell = zero_state
		out = M.LSTM(inp, hidden, cell, self.dim, self.name, self.reuse)
		self.reuse = True
		return out

a = tf.placeholder(tf.float32,[None,6,2])

a_split = tf.unstack(a,axis=1)

out_split = []

lstm = M.BasicLSTM(10,'LSTM1')
out = None
cell = None
for i in range(len(a_split)):
	out, cell = lstm.apply(a_split[i], out, cell)
	out_decoded = mod(out)
	out_split.append(out_decoded)

out = tf.stack(out_split,1) # should be in shape [None, 3, 1]

label_holder = tf.placeholder(tf.float32,[None,6,1])

# loss = tf.reduce_mean(tf.square(label_holder - out))
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out,labels=label_holder))

train_op = tf.train.AdamOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
	M.loadSess('./model/',sess,init=True)
	for i in range(10000):
		batch_x, batch_y = getInputVector()
		mask = np.ones([1,6,1])

		ls,_ = sess.run([loss,train_op],feed_dict={a:[batch_x], label_holder:[batch_y]})
		if i%1000==0:
			print(ls)
	test_x, test_y = getInputVector()
	o = sess.run(out,feed_dict={a:[test_x]})
	print(test_x)
	print(test_y)
	print(o)
	print(np.round(o))

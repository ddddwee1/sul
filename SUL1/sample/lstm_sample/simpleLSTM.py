import model as M 
import tensorflow as tf 
import numpy as np 

# ------ START data generator ---------
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

# --------- END data generator -----------

# --------- START post_layers ------------

def mod(inp,reuse=False):
	with tf.variable_scope('last',reuse=reuse):
		mod = M.Model(inp)
		mod.fcLayer(1)
	return mod.get_current_layer()

# --------- END post_layers --------------

a = tf.placeholder(tf.float32,[None,6,2])

lstm = M.SimpleLSTM(5, out_func=mod)
out = lstm.apply(a)

label_holder = tf.placeholder(tf.float32,[None,6,1])

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out,labels=label_holder))

train_op = tf.train.AdamOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
	M.loadSess('./model/',sess,init=True)
	for i in range(10000):
		batch_x, batch_y = getInputVector()
		ls,_ = sess.run([loss,train_op],feed_dict={a:[batch_x], label_holder:[batch_y]})
		if i%1000==0:
			print(ls)
	test_x, test_y = getInputVector()
	o = sess.run(out,feed_dict={a:[test_x]})
	print('Input\n',test_x)
	print('Label\n',test_y)
	print('Pred\n',np.int32(o>0))

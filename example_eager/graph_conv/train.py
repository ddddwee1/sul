import layers2 as L 
L.set_gpu('1')
import modeleag as M 
import tensorflow as tf 
import data_reader

class GCNmod(M.Model):
	def initialize(self, adj_mtx):
		self.c1 = M.GraphConvLayer(64, adj_mtx=adj_mtx, activation=M.PARAM_RELU)
		# self.c2 = M.GraphConvLayer(32, adj_mtx=adj_mtx, activation=M.PARAM_RELU)
		self.c3 = M.GraphConvLayer(7, adj_mtx=adj_mtx, usebias=False)

	def forward(self, x):
		x = self.c1(x)
		# x = self.c2(x)
		x = self.c3(x)
		return x 

def loss(outpt, indices, label, label_all):
	output_selected = tf.gather_nd(outpt, indices)
	ls = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=output_selected))
	correct = tf.equal(tf.cast(tf.argmax(outpt, -1), tf.int64), tf.cast(label_all, tf.int64))
	acc = tf.reduce_mean(tf.cast(correct, tf.float32))
	return ls, acc

reader = data_reader.data_reader()
reader.process_data()
features, adj, indices, label, label_all = reader.get_data()

mod = GCNmod(adj)
optim = tf.train.AdamOptimizer(0.01)

for i in range(100):
	with tf.GradientTape() as tape:
		out = mod(features)
		ls, acc = loss(out, indices, label, label_all)

	grad = tape.gradient(ls, mod.variables)
	optim.apply_gradients(zip(grad, mod.variables))

	print(i, '\t', ls.numpy(), '\t', acc.numpy())
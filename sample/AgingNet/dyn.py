import tensorflow as tf 
import model as M 
import os 
import layers as L

class model_attention(M.Model):
	def SelfAttention(self,att_num=None,is_fc=False,residual=False):
		assert is_fc or att_num, 'must state attention feature num for conv'
		def flatten_hw(layer):
			shape = layer.get_shape().as_list()
			layer = tf.reshape(layer,[-1,shape[1]*shape[2],shape[3]])
			return layer

		with tf.variable_scope('att_'+str(self.layernum)):
			# conv each of them
			current = self.result
			current_shape = current.get_shape().as_list()
			orig_num = current_shape[-1]
			if is_fc:
				f = L.Fcnn(current,orig_num,'att_fc_f'+str(self.layernum))
				g = L.Fcnn(current,orig_num,'att_fc_g'+str(self.layernum))
				h = L.Fcnn(current,orig_num,'att_fc_h'+str(self.layernum))
				f = tf.expand_dims(f,axis=-1)
				g = tf.expand_dims(g,axis=-1)
				h = tf.expand_dims(h,axis=-1)
			else:
				f = L.conv2D(current,1,att_num,'att_conv_f_'+str(self.layernum))
				g = L.conv2D(current,1,att_num,'att_conv_g_'+str(self.layernum))
				h = L.conv2D(current,1,orig_num,'att_conv_h_'+str(self.layernum))

				# flatten them
				f = flatten_hw(f)
				g = flatten_hw(g)
				h = flatten_hw(h)

			# softmax(fg)
			fg = tf.matmul(f,g,transpose_b=True)
			fg = tf.nn.softmax(fg,-1)

			# out = scale(softmax(fg)h) + x 
			scale = tf.Variable(0.)
			out = tf.matmul(fg,h)
			if is_fc:
				out = tf.reshape(out,[-1,orig_num])
			else:
				out = tf.reshape(out,[-1]+current_shape[1:3]+[orig_num])
			if residual:
				out = out + current
			self.layernum+=1
			self.inpsize = out.get_shape().as_list()
			self.result = out
		return self.result

	def res_block(self,ratio):
		current = self.result
		current_shape = current.get_shape().as_list()
		map_size = current_shape[-1]
		with tf.variable_scope('res_blk_'+str(self.layernum)):
			self.batch_norm()
			self.convLayer(1,map_size//ratio,activation=M.PARAM_RELU,batch_norm=True)
			self.convLayer(3,map_size//ratio,activation=M.PARAM_RELU,batch_norm=True)
			self.convLayer(1,map_size,activation=M.PARAM_RELU)
			# o = tf.reduce_max(tf.stack([current,self.result],axis=0),axis=0)
			self.result += current
		return self.result

	def QAttention(self,feature):
		with tf.variable_scope('Q_attention_'+str(self.layernum)):
			self.result = tf.expand_dims(self.result,-1) 
			e = tf.matmul(feature, self.result) # [bsize, feature_num, 1]
			e = tf.squeeze(e,[-1])
			e = tf.nn.softmax(e,-1)
			out = e * self.result
			out = tf.reduce_mean(out,1)
			self.result = out 
			self.inpsize = self.result.get_shape().as_list()
		return self.result

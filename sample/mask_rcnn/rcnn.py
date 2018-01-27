import tensorflow as tf 
import model as M
import layers as L
import utils2

class MaskRCNN():
	def __init__(self,config):
		self.config = config
		self.build_graph(config)

	def build_graph(self,config):
		h,w = config.IMAGE_SHAPE[:2]
		if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
			raise Exception("Image size must be dividable by 2 at least 6 times to avoid fractions when downscaling and upscaling. For example, use 256, 320, 384, 448, 512, ... etc. ")

		print(config.IMAGE_SHAPE.tolist())
		image_holder = tf.placeholder(tf.float32,[None]+config.IMAGE_SHAPE.tolist())
		meta_holder = tf.placeholder(tf.float32,[None])

		C1,C2,C3,C4,C5 = resnet(image_holder,stage5=True)
		P5 = L.conv2D(C5,1,256,'P5')
		P4 = L.upSampling(P5,2,'U5') + L.conv2D(C4,1,256,'P4')
		P3 = L.upSampling(P4,2,'U4') + L.conv2D(C3,1,256,'P3')
		P2 = L.upSampling(P3,2,'U3') + L.conv2D(C2,1,256,'P2')
		P2 = L.conv2D(P2,3,256,'P2_')
		P3 = L.conv2D(P3,3,256,'P3_')
		P4 = L.conv2D(P4,3,256,'P4_')
		P5 = L.conv2D(P5,3,256,'P5_')
		P6 = L.maxpooling(P5,1,2,'P6_')
		rpn_fm = [P2,P3,P4,P5,P6]
		mrcnn_fm = [P2,P3,P4,P5]

		self.anchors = utils2.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,config.RPN_ANCHOR_RATIOS,config.BACKBONE_SHAPES,config.BACKBONE_STRIDES,config.RPN_ANCHOR_STRIDE)

		rpn = rpn(config.RPN_ANCHOR_STRIDE,len(config.RPN_ANCHOR_RATIOS), 256)

		scale_rpns = []
		for p_layer in rpn_fm:
			scale_rpns.append(rpn.pred(p_layer))

		scale_rpns = list(zip(*scale_rpns))
		scale_rpns = [tf.concat(a,1) for a in scale_rpns]
		rpn_logits, rpn_class, rpn_bbox = scale_rpns

		proposal_count = config.POST_NMS_ROIS_INFERENCE

		# proposal_count,nms_threshold,anchors,config=None
		propLayer = proposal_Layer(proposal_count, config.RPN_NMS_THRESHOLD, self.anchors, config=config)

		rois = propLayer.apply([rpn_class, rpn_bbox])
		# TODO: add detection layer and mask layers




block_num = 0
def res_block(mod,kernel_size,channels,stride,with_short):
	chn1, chn2, chn3 = channels
	with tf.variable_scope('block_'+str(block_num)):
		inputLayer = mod.get_current()
		mod.convLayer(1,chn1,stride=stride,batch_norm=True,activation=M.PARAM_RELU)
		mod.convLayer(kernel_size,chn2,batch_norm=True,activation=M.PARAM_RELU)
		branch = mod.convLayer(1,chn3,batch_norm=True)
		# Shortcut
		mod.set_current(inputLayer)
		if with_short:
			mod.convLayer(1,chn3,stride=stride,batch_norm=True)
		mod.sum(branch)
		mod.activation(M.PARAM_RELU)
	block_num += 1

def resnet(inputLayer,stage5=False):
	with tf.variable_scope('resnet'):
		mod = M.Model(inputLayer,inputLayer.get_shape().as_list())
		# 1
		mod.padding(3)
		mod.convLayer(7,64,stride=2,pad='VALID',batch_norm=True,activation=M.PARAM_RELU)
		mod.maxpoolLayer(3,stride=2)
		C1 = mod.get_current_layer()
		# 2
		res_block(mod,3,[64,64,256],1,True)
		res_block(mod,3,[64,64,256],1,False)
		res_block(mod,3,[64,64,256],1,False)
		C2 = mod.get_current_layer()
		# 3
		res_block(mod,3,[128,128,512],2,True)
		res_block(mod,3,[128,128,512],1,False)
		res_block(mod,3,[128,128,512],1,False)
		res_block(mod,3,[128,128,512],1,False)
		C3 = mod.get_current_layer()
		# 4
		res_block(mod,3,[256,256,1024],2,True)
		for i in range(22):
			res_block(mod,3,[256,256,1024],1,False)
		C4 = mod.get_current_layer()
		# 5
		if stage5:
			mod.res_block(mod,3,[512,512,2048],2,True)
			mod.res_block(mod,3,[512,512,2048],1,False)
			mod.res_block(mod,3,[512,512,2048],1,False)
			C5 = mod.get_current_layer()
		else:
			C5 = None
	return mod,C1,C2,C3,C4,C5

class rpn():
	def __init__(self,anchor_stride,anchor_density,channels):
		self.anchor_stride = anchor_stride
		self.anchor_density = anchor_density
		self.channels = channels
		self.reuse=False

	def pred(self,inp):
		with tf.variable_scope('RPN',reuse=self.reuse):
			# input_holder = tf.placeholder(tf.float32,[None,None,None,channels])
			shared_feature = L.convLayer(inp,3,512,stride=self.anchor_stride,'rpn_shared')
			# logits
			rpn_logits = L.convLayer(shared_feature,1,2*self.anchor_density,'rpn_logits')
			rpn_logits = tf.reshape(rpn_logits,[rpn_logits.get_shape().as_list()[0],-1,2])
			rpn_prob = tf.nn.softmax(rpn_logits)
			# bbox
			rpn_bbox = L.convLayer(shared_feature,1,4*self.anchor_density,'rpn_bbox')
			rpn_bbox = tf.reshape(rpn_bbox,[rpn_bbox.get_shape().as_list()[0],-1,4])
			self.reuse=True
		return rpn_logits,rpn_prob,rpn_bbox

class proposal_Layer():
	def __init__(self,proposal_count,nms_threshold,anchors,config=None):
		self.config = config
		self.proposal_count = proposal_count
		self.nms_threshold = nms_threshold
		self.anchors = anchors.astype(np.float32)

	def apply(self,inputs):
		scores = inputs[0][:,:,1]
		deltas = inputs[1]
		deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1,1,4])
		anchors = self.anchors

		pre_nms_limit = min(6000, self.anchors.shape[0])

		ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True).indices
		scores = self.split_batch([scores,ix], lambda x,y: tf.gather(x,y), self.config.IMAGES_PER_GPU)
		deltas = self.split_batch([deltas,ix], lambda x,y: tf.gather(x,y), self.config.IMAGES_PER_GPU)
		anchors = self.split_batch([anchors,ix], lambda x,y: tf.gather(x,y), self.config.IMAGES_PER_GPU)

		boxes = self.split_batch([anchors,deltas], lambda x,y: self.apply_box_deltas_graph(x,y),self.config.IMAGES_PER_GPU)

		h,w = self.config.IMAGE_SHAPE[:2]

		window = np.array([0,0,h,w]).astype(np.float32)
		boxes = split_batch(boxes, lambda x: self.clip_boxes_graph(x,window),self.config.IMAGES_PER_GPU)

		normalized_boxes = boxes / np.array([[h,w,h,w]])

		proposals = self.split_batch([normalized_boxes, scores], self.nonMax, self.config.IMAGES_PER_GPU)
		return proposals


	def split_batch(self,inputs,graph,bsize):
		result = []
		for i in range(bsize):
			single_img = [k[i] for k in inputs]
			buff = graph(*single_img[i])
			result.append(buff)
		return result

	def apply_box_deltas_graph(self, boxes, deltas):
		"""Applies the given deltas to the given boxes.
		boxes: [N, 4] where each row is y1, x1, y2, x2
		deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
		"""
		# Convert to y, x, h, w
		height = boxes[:, 2] - boxes[:, 0]
		width = boxes[:, 3] - boxes[:, 1]
		center_y = boxes[:, 0] + 0.5 * height
		center_x = boxes[:, 1] + 0.5 * width
		# Apply deltas
		center_y += deltas[:, 0] * height
		center_x += deltas[:, 1] * width
		height *= tf.exp(deltas[:, 2])
		width *= tf.exp(deltas[:, 3])
		# Convert back to y1, x1, y2, x2
		y1 = center_y - 0.5 * height
		x1 = center_x - 0.5 * width
		y2 = y1 + height
		x2 = x1 + width
		result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
		return result

	def clip_boxes_graph(self, boxes, window):
		"""
		boxes: [N, 4] each row is y1, x1, y2, x2
		window: [4] in the form y1, x1, y2, x2
		"""
		# Split corners
		wy1, wx1, wy2, wx2 = tf.split(window, 4)
		y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
		# Clip
		y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
		x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
		y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
		x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
		clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
		return clipped

	def nonMax(self,normalized_boxes,scores):
		indices = tf.image.non_max_suppression(normalized_boxes, scores, self.proposal_count, self.nms_threshold, name="rpn_non_max_suppression")
		proposals = tf.gather(normalized_boxes, indices)
		# Pad if needed
		padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
		proposals = tf.pad(proposals, [(0, padding), (0, 0)])
		return proposals

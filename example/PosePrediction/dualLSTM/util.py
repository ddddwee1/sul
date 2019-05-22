import config 
import numpy as np 

BATCH_SIZE = config.BATCH_SIZE
SEQ_IN = config.SEQ_IN
SEQ_OUT = config.SEQ_OUT
IN_DIM = config.IN_DIM

def convert_velocity(data):
	velocity = data[1:] - data[:-1]
	return velocity

def get_batch(data, one_hot, actions):
	"""Get a random batch of data from the specified bucket, prepare for step.

	Args
		data: a list of sequences of size n-by-d to fit the model to.
		actions: a list of the actions we are using
	Returns
		The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
		the constructed batches have the proper format to call step(...) later.
	"""

	# Select entries at random
	all_keys		= list(data.keys())
	chosen_keys = np.random.choice( len(all_keys), BATCH_SIZE )

	# How many frames in total do we need?
	total_frames = SEQ_IN + SEQ_OUT

	encoder_inputs	= np.zeros((BATCH_SIZE, SEQ_IN-1, IN_DIM), dtype=float)
	decoder_inputs	= np.zeros((BATCH_SIZE, SEQ_OUT, IN_DIM), dtype=float)
	decoder_outputs = np.zeros((BATCH_SIZE, SEQ_OUT, IN_DIM), dtype=float)

	for i in range( BATCH_SIZE ):

		the_key = all_keys[ chosen_keys[i] ]

		# Get the number of frames
		n, _ = data[ the_key ].shape

		# Sample somewherein the middle
		idx = np.random.randint( 16, n-total_frames )

		# Select the data around the sampled points
		data_sel = data[ the_key ][idx:idx+total_frames ,:]

		# Add the data
		encoder_inputs[i,:,0:IN_DIM]	= data_sel[0:SEQ_IN-1, :]
		decoder_inputs[i,:,0:IN_DIM]	= data_sel[SEQ_IN-1:SEQ_IN+SEQ_OUT-1, :]
		decoder_outputs[i,:,0:IN_DIM] = data_sel[SEQ_IN:, 0:IN_DIM]

	return encoder_inputs, decoder_inputs, decoder_outputs 

def find_indices_srnn(data, action ):
	"""
	Find the same action indices as in SRNN.
	See https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
	"""

	# Used a fixed dummy seed, following
	# https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
	SEED = 1234567890
	rng = np.random.RandomState( SEED )

	subject = 5
	subaction1 = 1
	subaction2 = 2

	T1 = data[ (subject, action, subaction1, 'even') ].shape[0]
	T2 = data[ (subject, action, subaction2, 'even') ].shape[0]
	prefix, suffix = 50, 100

	idx = []
	idx.append( rng.randint( 16,T1-prefix-suffix ))
	idx.append( rng.randint( 16,T2-prefix-suffix ))
	idx.append( rng.randint( 16,T1-prefix-suffix ))
	idx.append( rng.randint( 16,T2-prefix-suffix ))
	idx.append( rng.randint( 16,T1-prefix-suffix ))
	idx.append( rng.randint( 16,T2-prefix-suffix ))
	idx.append( rng.randint( 16,T1-prefix-suffix ))
	idx.append( rng.randint( 16,T2-prefix-suffix ))
	return idx

def get_batch_srnn(data, action, actions ):
	"""
	Get a random batch of data from the specified bucket, prepare for step.

	Args
		data: dictionary with k:v, k=((subject, action, subsequence, 'even')),
			v=nxd matrix with a sequence of poses
		action: the action to load data from
	Returns
		The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
		the constructed batches have the proper format to call step(...) later.
	"""

	#actions = ["directions", "discussion", "eating", "greeting", "phoning",
	#					"posing", "purchases", "sitting", "sittingdown", "smoking",
	#					"takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]

	if not action in actions:
		raise ValueError("Unrecognized action {0}".format(action))

	frames = {}
	frames[ action ] = find_indices_srnn( data, action )

	batch_size = 8 # we always evaluate 8 seeds
	subject		= 5 # we always evaluate on subject 5
	source_seq_len = seq_length_in
	target_seq_len = seq_length_out

	seeds = [( action, (i%2)+1, frames[action][i] ) for i in range(batch_size)]

	encoder_inputs	= np.zeros( (batch_size, source_seq_len-1, input_size), dtype=float )
	decoder_inputs	= np.zeros( (batch_size, target_seq_len, input_size), dtype=float )
	decoder_outputs = np.zeros( (batch_size, target_seq_len, input_size), dtype=float )

	# Compute the number of frames needed
	total_frames = source_seq_len + target_seq_len

	# Reproducing SRNN's sequence subsequence selection as done in
	# https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L343
	for i in range( batch_size ):

		_, subsequence, idx = seeds[i]
		idx = idx + 50

		data_sel = data[ (subject, action, subsequence, 'even') ]

		data_sel = data_sel[(idx-source_seq_len):(idx+target_seq_len) ,:]

		encoder_inputs[i, :, :]	= data_sel[0:source_seq_len-1, :]
		decoder_inputs[i, :, :]	= data_sel[source_seq_len-1:(source_seq_len+target_seq_len-1), :]
		decoder_outputs[i, :, :] = data_sel[source_seq_len:, :]


	return encoder_inputs, decoder_inputs, decoder_outputs 

def get_srnn_gts( actions, test_set, data_mean, data_std, dim_to_ignore, one_hot, to_euler=True ):
	srnn_gts_euler = {}
	for action in actions:
		srnn_gt_euler = []
		_, _, srnn_expmap, = get_batch_srnn( test_set, action, actions )
		# expmap -> rotmat -> euler
		for i in np.arange( srnn_expmap.shape[0] ):
			denormed = data_utils.unNormalizeData(srnn_expmap[i,:,:], data_mean, data_std, dim_to_ignore, actions, one_hot )
			if to_euler:
				for j in np.arange( denormed.shape[0] ):
					for k in np.arange(3,97,3):
						denormed[j,k:k+3] = data_utils.rotmat2euler( data_utils.expmap2rotmat( denormed[j,k:k+3] ))
			srnn_gt_euler.append( denormed );
		# Put back in the dictionary
		srnn_gts_euler[action] = srnn_gt_euler
	return srnn_gts_euler

import data_utils
import os 
import numpy as np 
import pickle 

actions = ["walking", "eating", "smoking", "discussion",  "directions",
              "greeting", "phoning", "posing", "purchases", "sitting",
              "sittingdown", "takingphoto", "waiting", "walkingdog",
              "walkingtogether"]
seq_length_in = 50
seq_length_out = 25
input_size = 54
data_dir = os.path.normpath("./data/h3.6m/dataset")

def read_all_data( actions, seq_length_in, seq_length_out, data_dir, one_hot ):
  """
  Loads data for training/testing and normalizes it.

  Args
    actions: list of strings (actions) to load
    seq_length_in: number of frames to use in the burn-in sequence
    seq_length_out: number of frames to use in the output sequence
    data_dir: directory to load the data from
    one_hot: whether to use one-hot encoding per action
  Returns
    train_set: dictionary with normalized training data
    test_set: dictionary with test data
    data_mean: d-long vector with the mean of the training data
    data_std: d-long vector with the standard dev of the training data
    dim_to_ignore: dimensions that are not used becaused stdev is too small
    dim_to_use: dimensions that we are actually using in the model
  """

  # === Read training data ===
  print ("Reading training data (seq_len_in: {0}, seq_len_out {1}).".format(
           seq_length_in, seq_length_out))

  train_subject_ids = [1,6,7,8,9,11]
  test_subject_ids = [5]

  train_set, complete_train = data_utils.load_data( data_dir, train_subject_ids, actions, one_hot )
  test_set,  complete_test  = data_utils.load_data( data_dir, test_subject_ids,  actions, one_hot )

  # Compute normalization stats
  data_mean, data_std, dim_to_ignore, dim_to_use = data_utils.normalization_stats(complete_train)

  # Normalize -- subtract mean, divide by stdev
  train_set = data_utils.normalize_data( train_set, data_mean, data_std, dim_to_use, actions, one_hot )
  test_set  = data_utils.normalize_data( test_set,  data_mean, data_std, dim_to_use, actions, one_hot )
  print("done reading data.")

  return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use

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
  #          "posing", "purchases", "sitting", "sittingdown", "smoking",
  #          "takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]

  if not action in actions:
    raise ValueError("Unrecognized action {0}".format(action))

  frames = {}
  frames[ action ] = find_indices_srnn( data, action )

  batch_size = 8 # we always evaluate 8 seeds
  subject    = 5 # we always evaluate on subject 5
  source_seq_len = seq_length_in
  target_seq_len = seq_length_out

  seeds = [( action, (i%2)+1, frames[action][i] ) for i in range(batch_size)]

  encoder_inputs  = np.zeros( (batch_size, source_seq_len-1, input_size), dtype=float )
  decoder_inputs  = np.zeros( (batch_size, target_seq_len, input_size), dtype=float )
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

    encoder_inputs[i, :, :]  = data_sel[0:source_seq_len-1, :]
    decoder_inputs[i, :, :]  = data_sel[source_seq_len-1:(source_seq_len+target_seq_len-1), :]
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



# data = read_all_data(actions, seq_length_in, seq_length_out, data_dir, False)
# pickle.dump(data, open('data.pkl','wb'))
data = pickle.load(open('data.pkl','rb'))
train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = data 

gts_expmap = get_srnn_gts(actions, test_set, data_mean, data_std, dim_to_ignore, False, False)
gts_euler = get_srnn_gts(actions, test_set, data_mean, data_std, dim_to_ignore, False)

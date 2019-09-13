raw_path = './LJSpeech/'
data_path = './data/'

melpath = './data/mel/'
quantpath = './data/quant/'

voc_mode = 'MOL'
extension = '.wav'
ignore_tts = False
norm_peaks = True
mu_law = True

sample_rate = 22050
n_fft = 2048
hop_length = 275 
win_length = 1100  
min_level_db = -100
ref_level_db = 20
num_mels = 80
fmin = 40

# GTA PARAM
gta_path = './gta/'
model_folder = './model_gta/'
schedule = [(7,  1e-3,  10_000,  32),   # progressive training schedule
			(5,  1e-4, 100_000,  32),   # (r, lr, step, batch_size)
			(2,  1e-4, 180_000,  16),
			(1,  1e-4, 350_000,  8)]

tts_cleaner_names = ['english_cleaners']

# Model Hparams
tts_r = 1                           # model predicts r frames per output step
tts_embed_dims = 256                # embedding dimension for the graphemes/phoneme inputs
tts_encoder_dims = 128
tts_decoder_dims = 256
tts_postnet_dims = 128
tts_encoder_K = 16
tts_lstm_dims = 512
tts_postnet_K = 8
tts_num_highways = 4
tts_dropout = 0.5
tts_cleaner_names = ['english_cleaners']
tts_max_mel_len = 1250              # if you have a couple of extremely long spectrograms you might want to use this
tts_bin_lengths = True              # bins the spectrogram lengths before sampling in data loader - speeds up training
tts_clip_grad_norm = 1.0            # clips the gradient norm to prevent explosion - set to None if not needed
tts_checkpoint_every = 2_000        # checkpoints the model every X steps
# TODO: tts_phoneme_prob = 0.0              # [0 <-> 1] probability for feeding model phonemes vrs graphemes

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
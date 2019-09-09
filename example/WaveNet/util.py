import glob 
import librosa 
import config 
import numpy as np 

mel_basis = None

def stft(y):
	return librosa.stft(y=y, n_fft=config.n_fft, hop_length=config.hop_length, win_length=config.win_length)

def amp_to_db(x):
	return 20 * np.log10(np.maximum(1e-5, x))

def db_to_amp(x):
	return np.power(10.0, x * 0.05)

def build_mel_basis():
	return librosa.filters.mel(config.sample_rate, config.n_fft, n_mels=config.num_mels, fmin=config.fmin)

def linear_to_mel(spectrogram):
	global mel_basis
	if mel_basis is None:
		mel_basis = build_mel_basis()
	return np.dot(mel_basis, spectrogram)

def normalize(S):
	return np.clip((S - config.min_level_db) / -config.min_level_db, 0, 1)

def denormalize(S):
	return (np.clip(S, 0, 1) * -config.min_level_db) + config.min_level_db

def load_wav(path):
	return librosa.load(path, sr=config.sample_rate)[0]

def spectrogram(y):
	D = stft(y)
	S = amp_to_db(np.abs(D)) - config.ref_level_db
	return normalize(S)

def melspectrogram(y):
	D = stft(y)
	S = amp_to_db(linear_to_mel(np.abs(D)))
	return normalize(S)

def encode_mu_law(x, mu):
	mu = mu - 1
	fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
	return np.floor((fx + 1) / 2 * mu + 0.5)

def decode_mu_law(y, mu, from_labels=True):
	# TODO: get rid of log2 - makes no sense
	if from_labels: y = label_2_float(y, math.log2(mu))
	mu = mu - 1
	x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
	return x

def float2label(y, bits):
	y = (y + 1.) * (2**bits - 1) /2 
	return y.clip(0, 2**bits -1)

def label2float(x, bits):
	return 2*x / (2**bits -1.) -1.

def get_files(path, extension):
	fnames = glob.glob(f'{path}/**/*{extension}', recursive=True)
	return fnames


#### text dict ####
def ljspeech(path) :
	csv_file = get_files(path, extension='.csv')
	assert len(csv_file) == 1
	text_dict = {}
	with open(csv_file[0], encoding='utf-8') as f :
		for line in f :
			split = line.split('|')
			text_dict[split[0]] = split[-1]
	return text_dict

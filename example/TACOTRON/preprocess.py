import util 
import config 
import numpy as np 
import pickle 
import os 

extension = config.extension
path = config.raw_path 

def convert_file(path):
	y = util.load_wav(path)
	peak = np.abs(y).max()
	if config.norm_peaks or peak>1.0:
		y /= peak
	mel = util.melspectrogram(y)
	if config.voc_mode == 'RAW':
		quant = util.encode_mu_law(y, mu=2**config.bits) if config.mu_law else util.float2label(y, bits=config.bits)
	elif config.voc_mode == 'MOL':
		quant = util.float2label(y, bits=16)
	return np.float32(mel), np.int64(quant)

def process_wav(path):
	path = path.replace('\\','/')
	idd = path.split('/')[-1][:-4] # hardcode here
	m, x = convert_file(path)
	np.save(f'{config.melpath}{idd}.npy', m, allow_pickle=False)
	np.save(f'{config.quantpath}{idd}.npy', x, allow_pickle=False)
	return idd, m.shape[-1]

for i in [config.data_path, config.melpath, config.quantpath]:
	if not os.path.exists(i):
		os.makedirs(i)

wav_files = util.get_files(path, extension)
print(f'{len(wav_files)} files found in {path}')

if not config.ignore_tts:
	text_dict = util.ljspeech(path)
	with open(f'{config.data_path}text_dict.pkl', 'wb') as f:
		pickle.dump(text_dict, f)

dataset = []
for i, wf in enumerate(wav_files):
	idd, length = process_wav(wf)
	dataset += [(idd, length)]
	print(f'{i}/{len(wav_files)}')
with open(f'{config.data_path}dataset.pkl', 'wb') as f:
	pickle.dump(dataset, f)
print('Complete pre-processing.')

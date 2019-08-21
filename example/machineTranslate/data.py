import unicodedata
import model3 as M 
import re
import numpy as np 
import random 

max_length = 20

class Language():
	def __init__(self, name):
		self.name = name 
		self.w2i = {}
		self.counter = {}
		self.i2w = {0: '<SOS>', 1: '<EOS>', 2: '<UNK>'}
		self.n_words = len(self.i2w)
	def push_word(self, w):
		if not w in self.w2i:
			self.w2i[w] = self.n_words
			self.counter[w] = 1
			self.i2w[self.n_words] = w 
			self.n_words += 1
		else:
			self.counter[w] += 1
	def push_sentence(self, s):
		for w in s.split(' '):
			self.push_word(w)
	def encode_sentence(self, s):
		codes = [self.w2i[w] for w in s.split(' ')]
		codes.append(1)
		return codes 
	def decode_sentence(self, code):
		words = [self.i2w[c] for c in codes]
		return ' '.join(words)
	def finish(self):
		self.eye = np.eye(self.n_words)

def is_valid_length(p):
	return len(p[0].split(' '))<max_length and len(p[1].split(' '))<max_length

def unicode_to_ascii(s):
	chars = [c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn']
	char_list = ''.join(chars)
	return char_list

def normalize_string(s):
	s = unicode_to_ascii(s.lower().strip())
	s = re.sub(r"([.!?])", r" \1", s)
	s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
	return s

def read_language(lang):
	doc = open('./%s.txt'%lang, encoding='utf-8').read()
	lines = doc.strip().split('\n')
	pairs = [[normalize_string(s) for s in line.split('\t')] for line in lines]
	pairs = [p for p in pairs if is_valid_length(p)]
	eng = Language('eng')
	lan = Language(lang)
	for p in pairs:
		eng.push_sentence(p[0])
		lan.push_sentence(p[1])
	eng.finish()
	lan.finish()
	return eng, lan, pairs 

class data_reader(M.ThreadReader):
	def _get_data(self):
		self.eng, self.lan, data = read_language(self.lang)
		return data 

	def _next_iter(self):
		return [random.choice(self.data)]
		
	def _process_data(self, item):
		src = item[0]
		tgt = item[1]
		src_code = self.eng.encode_sentence(src)
		tgt_code = self.lan.encode_sentence(tgt)
		src_code = self.eng.eye[np.int32(src_code)]
		tgt_code = self.lan.eye[np.int32(tgt_code)]
		return src_code, tgt_code

if __name__=='__main__':
	reader = data_reader(1, lang='fra')
	batch = reader.get_next()
	print(batch[0])
	print(batch[0][0].shape)
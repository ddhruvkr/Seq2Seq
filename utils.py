from __future__ import unicode_literals, print_function, division
import time
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 10
SOS_token = 0
EOS_token = 1

eng_prefixes = (
	"i am ", "i m ",
	"he is", "he s ",
	"she is", "she s ",
	"you are", "you re ",
	"we are", "we re ",
	"they are", "they re "
)


class Lang:
	def __init__(self, name):
		self.name = name
		self.word2index = {}
		self.word2count = {}
		self.index2word = {1: "SOS", 2: "EOS", 0: "PAD"}
		self.n_words = 3  # Count SOS and EOS

	def addSentence(self, sentence):
		for word in sentence.split(' '):
			self.addWord(word)

	def addWord(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.n_words
			self.word2count[word] = 1
			self.index2word[self.n_words] = word
			self.n_words += 1
		else:
			self.word2count[word] += 1

	# Turn a Unicode string to plain ASCII, thanks to
	# https://stackoverflow.com/a/518232/2809427
	def unicodeToAscii(s):
		return ''.join(
		c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn'
		)

	# Lowercase, trim, and remove non-letter characters


	def normalizeString(s):
		s = Lang.unicodeToAscii(s.lower().strip())
		s = re.sub(r"([.!?])", r" \1", s)
		s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
		return s


	def readLangs(lang1, lang2, reverse=False):
		print("Reading lines...")

		# Read the file and split into lines
		lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
			read().strip().split('\n')

		# Split every line into pairs and normalize
		pairs = [[Lang.normalizeString(s) for s in l.split('\t')] for l in lines]

		# Reverse pairs, make Lang instances
		if reverse:
			pairs = [list(reversed(p)) for p in pairs]
			input_lang = Lang(lang2)
			output_lang = Lang(lang1)
		else:
			input_lang = Lang(lang1)
			output_lang = Lang(lang2)

		return input_lang, output_lang, pairs

def filterPair(p):
	return len(p[0].split(' ')) < MAX_LENGTH and \
		len(p[1].split(' ')) < MAX_LENGTH
		#p[1].startswith(eng_prefixes)


def filterPairs(pairs):
	return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False):
	input_lang, output_lang, pairs = Lang.readLangs(lang1, lang2, reverse)
	print("Read %s sentence pairs" % len(pairs))
	pairs = filterPairs(pairs)
	print("Trimmed to %s sentence pairs" % len(pairs))
	print("Counting words...")
	for pair in pairs:
		input_lang.addSentence(pair[0])
		output_lang.addSentence(pair[1])
	print("Counted words:")
	print(input_lang.name, input_lang.n_words)
	print(output_lang.name, output_lang.n_words)
	return input_lang, output_lang, pairs


def pad_sequences(x, max_len):
    padded = np.zeros((max_len), dtype=np.int64)
    if len(x) > max_len: padded[:] = x[:max_len]
    else:
    	print(padded)
    	print(x)
    	padded[:len(x)] = x
    return padded


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair, input_lang, output_lang):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

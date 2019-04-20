from utils import *
from model.encoder import EncoderRNN
from model.decoder import DecoderRNN
from model.decoder_attn import AttnDecoderRNN
from test import *
from evaluate import *

hidden_size = 256

input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))


encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
#decoder1 = DecoderRNN(hidden_size, output_lang.n_words).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

trainIters(encoder1, attn_decoder1, 75000, pairs, input_lang, output_lang, print_every=5000)

evaluateRandomly(encoder1, attn_decoder1)
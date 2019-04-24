import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_forcing_ratio = 0.75
# TODO: decrease this ratio as training proceeds

def train(batch_size, input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden(batch_size, False)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    input_length = input_tensor.size(1)
    target_length = max_length

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0
    encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)
    decoder_input = torch.full((batch_size, 1), SOS_token, device=device, dtype=torch.int64)
    decoder_hidden = encoder_hidden
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_output)
            indices = torch.tensor([di], device=device)
            target_t = torch.index_select(target_tensor, 1, indices)
            loss += criterion(decoder_output[0], target_t.view(-1))
            decoder_input = target_t  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_output)
            indices = torch.tensor([di], device=device)
            target_t = torch.index_select(target_tensor, 1, indices)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze(0).detach()  # detach from history as input
            loss += criterion(decoder_output[0], target_t.view(-1))
            #if decoder_input.item() == EOS_token:
            #    break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

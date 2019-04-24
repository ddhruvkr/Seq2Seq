import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

teacher_forcing_ratio = 0.0
from utils import *

def train(batch_size, input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden(batch_size, False)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    #print(input_tensor)
    #print(input_tensor.shape)
    #print(target_tensor)
    input_length = input_tensor.size(1)
    target_length = max_length

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0
    #print(input_tensor)
    #print(input_tensor[0][ei])
    encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)
    #print(encoder_output)
    #print(encoder_hidden)
    #print(encoder_output[0,0])
    #encoder_outputs[ei] = encoder_output[0, 0]

    #decoder_input = torch.tensor([[SOS_token],[SOS_token]], device=device)
    decoder_input = torch.full((batch_size, 1), SOS_token, device=device, dtype=torch.int64)
    #(batch_size, start_token=1)
    #print(decoder_input.shape) #[2,1]
    decoder_hidden = encoder_hidden
    #print("target tensor")
    #print(target_tensor.shape) #[2,5]
    #print(target_tensor)
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            #print("use teacher forcing")
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_output, False)
            #print("decoder_output[0]")
            #print(decoder_output[0].shape)
            #print(target_tensor)
            indices = torch.tensor([di], device=device)
            target_t = torch.index_select(target_tensor, 1, indices)
            #print("target_t")
            #print(target_t)
            #print(target_t.view(-1))
            #print(target_t.shape)
            #print("use teacher forcing")
            #print(decoder_output)
            #print(target_t.view(-1))
            loss += criterion(decoder_output[0], target_t.view(-1))
            #print("check")
            #print(target_t.unsqueeze(1).shape)
            decoder_input = target_t  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            #print("dont use teacher forcing")
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_output, False)
            #print("decoder_output[0]")
            #print(decoder_output[0].shape)
            #print(target_tensor)
            #print(target_tensor)
            indices = torch.tensor([di], device=device)
            target_t = torch.index_select(target_tensor, 1, indices)
            #print("target_t")
            #print(target_t)
            #print(target_t.view(-1))
            #print(target_t.shape)
            topv, topi = decoder_output.topk(1)
            #print("topv")
            #print(topv.shape)
            #print("topi")
            #print(topi.shape)
            #print(topi)
            decoder_input = topi.squeeze(0).detach()  # detach from history as input
            #print("decoder input")
            #print(decoder_input.shape)
            #print(decoder_input)
            #print("decoder output")
            #print(decoder_output.shape)
            #indices = torch.tensor([di])
            #target_t = torch.index_select(target_tensor, 1, indices)
            #print(target_t.shape)
            #print(target_tensor.shape)
            #print("dont ues teacher forcing")
            #print(decoder_output)
            #print(target_t.view(-1))
            loss += criterion(decoder_output[0], target_t.view(-1))
            #if decoder_input.item() == EOS_token:
            #    break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

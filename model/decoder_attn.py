import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers = 4)
        self.gru1 = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs, pr):
        if pr:
            print("Start")
        #embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        #print(hidden)
        if pr:
            print(embedded.shape)
            print(hidden.shape)
        #output, hidden = self.gru1(embedded, hidden)
        #print(encoder_outputs.shape)
        #print(output.shape)
        #print(hidden.shape)
        #output_t = torch.transpose(output, 1, 2)
        #print(output_t.shape)
        #print(encoder_outputs.shape) #max_length, batch_size, dim
        #print(hidden.shape)
        #print(hidden[-1].shape) #batch_size, dim, 1
        encoder_outputs_s = encoder_outputs.transpose(0,1) #batch_size, max_len, dim
        if pr:
            print(encoder_outputs_s.shape)
            #h1 = hidden[-1].unsqueeze[0]
            print(hidden[-1].unsqueeze(2).shape)
        score = torch.bmm(encoder_outputs_s, hidden[-1].unsqueeze(2))
        #print(score)
        if pr:
            print(score.shape) #bts, max_len, 1
        attn_weights = F.softmax(score, dim=1)
        #print('my')
        #print(attn_weights)
        #attn_weights = F.softmax(
        #    self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        #print('org')
        #print(attn_weights.shape)
        #(bts, dim, max_len)(bts, max_len, 1)
        attn_applied = torch.bmm(encoder_outputs_s.transpose(1,2),attn_weights)
        #print(attn_applied)
        if pr:
            print(embedded.shape) #bts, 1, dim
            print(attn_applied.shape) #bts, dim, 1
        output = torch.cat((embedded, attn_applied.transpose(1,2)), 2)
        #print(output.shape) #bts, 1, 2dim
        #output = torch.cat((hidden, attn_applied), 2)
        output = self.attn_combine(output)
        #print(output)
        #print("output going to gru")
        #print(output.shape)
        output = F.relu(output)
        output, hidden = self.gru(output.transpose(0,1), hidden)
        #print(output)
        #print("output from gru")
        #print(output[0].shape)
        output = self.out(output)
        #print(output)
        #print(output.shape)
        output = F.log_softmax(output, dim=2)
        #print(output)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(4, 10, self.hidden_size, device=device)

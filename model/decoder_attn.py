import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, num_layers, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers = self.num_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        #print(encoder_outputs.shape) #max_length, batch_size, dim
        #print(hidden[-1].shape) #batch_size, dim, 1
        encoder_outputs_s = encoder_outputs.transpose(0,1) #batch_size, max_len, dim

        score = torch.bmm(encoder_outputs_s, hidden[-1].unsqueeze(2))
        #print(score.shape) #bts, max_len, 1
        attn_weights = F.softmax(score, dim=1)
        #(bts, dim, max_len)(bts, max_len, 1)
        attn_applied = torch.bmm(encoder_outputs_s.transpose(1,2),attn_weights)
        #print(embedded.shape) #bts, 1, dim
        #print(attn_applied.shape) #bts, dim, 1
        output = torch.cat((embedded, attn_applied.transpose(1,2)), 2)
        #print(output.shape) #bts, 1, 2*dim
        output = self.attn_combine(output)
        output = F.relu(output)
        output, hidden = self.gru(output.transpose(0,1), hidden)
        output = self.out(output)
        output = F.log_softmax(output, dim=2)
        return output, hidden, attn_weights

    def initHidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

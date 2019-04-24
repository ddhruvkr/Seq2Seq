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

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        #print(embedded.shape)
        #print(hidden.shape)
        #output, hidden = self.gru1(embedded, hidden)
        #print(encoder_outputs.shape)
        #print(output.shape)
        #print(hidden.shape)
        #output_t = torch.transpose(output, 1, 2)
        #print(output_t.shape)
        encoder_outputs_s = encoder_outputs.transpose(0,1).unsqueeze(0)
        #print(encoder_outputs_s.shape)
        #h1 = hidden[-1].unsqueeze[0]
        #print(hidden[-1].unsqueeze(0).shape)
        score = torch.bmm(hidden[-1].unsqueeze(0), encoder_outputs_s).squeeze(0)
        #print(score.shape)
        attn_weights = F.softmax(score, dim=1)
        #print('my')
        #print(attn_weights.shape)
        #attn_weights = F.softmax(
        #    self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        #print('org')
        #print(attn_weights.shape)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        #print(embedded.shape)
        #print(attn_applied.shape)
        output = torch.cat((embedded, attn_applied), 2)
        #output = torch.cat((hidden, attn_applied), 2)
        output = self.attn_combine(output)
        print(output.shape)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(4, 1, self.hidden_size, device=device)

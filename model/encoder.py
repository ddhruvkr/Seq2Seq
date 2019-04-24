import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers = self.num_layers)

    def forward(self, input, hidden):
        embedded = self.embedding(input).permute(1, 0, 2)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self, batch_size, evaluate):
        if evaluate:
            return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)
        else:
            return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

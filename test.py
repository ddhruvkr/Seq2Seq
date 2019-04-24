import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils import data
from utils import *
from train import *

class Dataset(data.Dataset):
    #'Characterizes a dataset for PyTorch'
    def __init__(self, x_pair, y_pair):
        #'Initialization'
        self.x_train = x_pair
        self.y_train = y_pair

    def __len__(self):
        #'Denotes the total number of samples'
        return len(self.x_train)

    def __getitem__(self, index):
        #'Generates one sample of data'
        # Select sample

        # Load data and get label
        x = self.x_train[index]
        y = self.y_train[index]

        return x, y

def load_data(dataset, batch_size):
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    return dataloader

def trainIters(encoder, decoder, n_iters, pairs, input_lang, output_lang, print_every=10, plot_every=100, learning_rate=0.01):
    start = time.time()
    batch_size = 1
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    #print(pairs)
    x_pair = []
    y_pair = []
    for i in range(len(pairs)):
        pair = tensorsFromPair(pairs[i], input_lang, output_lang)
        x_pair.append(pair[0])
        y_pair.append(pair[1])
    #print(x_pair.shape)
    #print(y_pair.shape)
    #print(x_pair[0])
    #print(pairs[0])
    x_pair = [pad_sequences(x, MAX_LENGTH) for x in x_pair]
    #print("padded x")
    #print(x_pair)
    y_pair = [pad_sequences(x, MAX_LENGTH) for x in y_pair]
    training_set = Dataset(x_pair, y_pair)
    training_iterator = load_data(training_set, batch_size)
    #training_pairs = [tensorsFromPair(random.choice(pairs), input_lang, output_lang)
    #                  for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for epoch in range(20):
        print('epocj')
        print(epoch)
        i = 0
        for input_tensor, target_tensor in training_iterator:
            #print(input_tensor)
            #print(target_tensor)
            loss = train(batch_size, input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
            i += 1
            #print(i)
            print_loss_total += loss
            plot_loss_total += loss

            if i % 100 == 0:
                print(i)
                print_loss_avg = print_loss_total / 100
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, (epoch+1) / 100),
                                             epoch+1, (epoch+1) / 100 * 100, print_loss_avg))

            if i % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        #showPlot(plot_losses)









    '''for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)'''

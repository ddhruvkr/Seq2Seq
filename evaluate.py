from utils import *

def evaluate(encoder, decoder, sentence, input_lang, output_lang, max_length=MAX_LENGTH):
    with torch.no_grad():
        #x_pair = []
        #y_pair = []
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        #pair = tensorsFromPair(sentence, input_lang, output_lang)
        #print(sentence)
        #print(pair)
        #x_pair.append(pair[0])
        #y_pair.append(pair[1])
        '''for i in range(len(sentence)):
            pair = tensorsFromPair(pairs[i], input_lang, output_lang)
            x_pair.append(pair[0])
            y_pair.append(pair[1])'''
        #print(x_pair.shape)
        #print(y_pair.shape)
        #print(x_pair[0])
        #print(pairs[0])
        #x_pair = [pad_sequences(x, MAX_LENGTH) for x in x_pair]
        #print("padded x")
        #print(x_pair)
        #y_pair = [pad_sequences(x, MAX_LENGTH) for x in y_pair]
        #training_set = Dataset(x_pair, y_pair)
        #training_iterator = load_data(training_set, batch_size)


        #input_tensor = tensorFromSentence(input_lang, sentence)
        #input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden(1, True)

        #encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        #print(input_tensor.shape)
        encoder_output, encoder_hidden = encoder(input_tensor.unsqueeze(0), encoder_hidden)

        


        '''for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]'''
        batch_size = 1
        decoder_input = torch.full((batch_size, 1), SOS_token, device=device, dtype=torch.int64)
        #(batch_size, start_token=1)
        #print(decoder_input.shape) #[2,1]
        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)
        print(decoder_input.shape)
        print(decoder_hidden.shape)
        print(encoder_output.shape)
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, False)
            #decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
            decoder_input = topi.squeeze(0).detach()
            #decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
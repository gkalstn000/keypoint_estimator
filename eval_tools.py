import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import random
import torch

import utils


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # 주기적인 간격에 이 locator가 tick을 설정
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
class Evaler :
    def __init__(self,
                 encoder,
                 decoder,
                 pairs,
                 max_length,
                 input_lang,
                 output_lang,
                 device,
                 SOS_token,
                 EOS_token,):
        self.encoder = encoder
        self.decoder = decoder
        self.pairs = pairs
        self.max_length = max_length
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.device = device
        self.SOS_token = SOS_token
        self.EOS_token = EOS_token

    def evaluateRandomly(self,n=10):
        for i in range(n):
            pair = random.choice(self.pairs)
            print('>', pair[0])
            print('=', pair[1])
            output_words, attentions = self.evaluate(sentence=pair[0])
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('')

    def evaluate(self, sentence):
        with torch.no_grad():
            input_tensor = utils.tensorFromSentence(lang = self.input_lang,
                                                    sentence=sentence,
                                                    EOS_token=self.EOS_token,
                                                    device=self.device)
            input_length = input_tensor.size()[0]
            encoder_hidden = self.encoder.initHidden()

            encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=self.device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[ei],
                                                         encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[self.SOS_token]], device=self.device)  # SOS

            decoder_hidden = encoder_hidden

            decoded_words = []
            decoder_attentions = torch.zeros(self.max_length, self.max_length)

            for di in range(self.max_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == self.EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(self.output_lang.index2word[topi.item()])

                decoder_input = topi.squeeze().detach()

            return decoded_words, decoder_attentions[:di + 1]
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
                 max_length,
                 dataloader,
                 device):
        self.encoder = encoder
        self.decoder = decoder
        self.max_length = max_length
        self.device = device
        self.dataloader = dataloader

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


from data.mydata import MyDataSet
import torch.utils.data as Data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.encoder import EncoderRNN, DecoderRNN
from models.network import AttnDecoderRNN
if __name__ == '__main__' :
    MAX_LENGTH = 19

    height = 256
    width = 256

    input_size = 2
    output_size = 2
    hidden_size = 10
    batch_size = 128

    teacher_forcing_ratio = 0.5

    data_path = 'dataset/train/pose_label.pkl'
    data_dict = utils.load_train_data(data_path)
    mydata = MyDataSet(data_dict, height, width)

    dataloader = Data.DataLoader(mydata, batch_size, True)

    encoder1 = EncoderRNN(input_size, hidden_size, device).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_size, MAX_LENGTH, device, dropout_p=0.1).to(device)

    eval = Evaler(encoder=encoder1,
                  decoder=attn_decoder1,
                  max_length=MAX_LENGTH,
                  dataloader=dataloader,
                  device=device)

    eval.evaluateRandomly()
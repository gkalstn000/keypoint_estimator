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

    def print_points(self, src, tgt, pred, key_point_name):
        src = src[:-1]
        tgt = tgt[1:]
        pred = pred.numpy()
        strFormat = '%-12s%-12s%-12s%-12s\n'
        strOut = strFormat % ('Name', 'Source', 'Target', 'Pred')
        for name, src_p, tgt_p, pred_ in zip(key_point_name, src.astype(np.int8), tgt.astype(np.int8), pred.astype(np.int8)) :
            strOut += strFormat % (name, src_p, tgt_p, pred_)
        print(strOut)
    def evaluateRandomly(self,n=10):
        key_point_name = ['Nose', 'Neck', 'R_shoulder', 'R_elbow', 'R_wrist', 'L_shoulder', 'L_elbow', 'L_wrist',
                          'R_pelvis', 'R_knee', 'R_ankle', 'L_pelvis', 'L_knee', 'L_ankle','R_eye', 'L_eye', 'R_ear', 'L_ear']
        for i in range(n):
            src, tgt = random.choice(self.dataloader.dataset)
            output_point, attentions = self.evaluate(point=src)
            self.print_points(src, tgt, output_point, key_point_name)

    def evaluate(self, point):
        with torch.no_grad():
            point = torch.from_numpy(point).float()
            encoder_hidden = self.encoder.initHidden()

            encoder_output, encoder_hidden = self.encoder(point, encoder_hidden)

            decoder_hidden = encoder_hidden

            decoded_outputs = []
            decoder_attentions = torch.zeros(self.max_length, self.max_length)

            decoder_input = point[0][None, None, :]
            for di in range(1, self.max_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_output)
                decoder_attentions[di-1] = decoder_attention
                decoder_input = decoder_output
                decoded_outputs.append(decoder_output)

            return torch.cat(decoded_outputs, 0).squeeze(), decoder_attentions


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
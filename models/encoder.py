import torch.nn as nn
import torch
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        # input_size : 2
        # hidden_size : 10
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size)
        self.device = device

    def forward(self, input, hidden):
        # input size : (L, N, H_in)
        if len(input.size()) == 2 :
            input = input.unsqueeze(1) # (L, N, H_in)
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device):
        # hidden_size : 10
        # output_size : 2
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.device = device

    def forward(self, input, hidden):
        output = F.relu(input)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)

import utils
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__' :
    data_path = 'dataset/train/pose_label.pkl'
    data = utils.load_train_data(data_path)

    key_points = data['fasionWOMENDressesid0000041606_7additional']

    input_ = torch.Tensor(key_points)

    encoder = EncoderRNN(2, 10, device)
    encoder_hidden = encoder.initHidden()
    output, hidden = encoder(input_, encoder_hidden)

    decoder = DecoderRNN(10, 2, device)
    decoder_hidden = decoder.initHidden()
    output, hidden = decoder(output, decoder_hidden)
import torch.nn as nn
import torch
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size)
        self.device = device

    def forward(self, input, hidden):
        output = input # (B, N, 2)
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.device = device

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
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

    input_ = torch.Tensor(key_points).unsqueeze(0)

    encoder = EncoderRNN(2, 10, device)
    encoder_hidden = encoder.initHidden()
    output, hidden = encoder(input_, encoder_hidden)
import torch.nn as nn
import torch
import torch.nn.functional as F

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, device, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Linear(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.device = device

    def forward(self, input, hidden, encoder_outputs):
        # input size : (B, 1, 2)
        # hidden size : (1, B, hidden)
        # embedded size : (B, 1, hidden)
        embedded = self.embedding(input) # (B, L, hidden)
        embedded = self.dropout(embedded)

        hidden_transpose = hidden.transpose(1, 0)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded, hidden_transpose), dim = 2)), dim=2) # (B, 1, L)
        attn_applied = torch.bmm(attn_weights, encoder_outputs)

        output = torch.cat((embedded, attn_applied), 2)
        output = self.attn_combine(output)

        # output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = self.out(output)
        return output, hidden, attn_weights

    def initHidden(self, batch_size = 1):
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)

import utils
from models.encoder import EncoderRNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__' :
    data_path = 'dataset/train/pose_label.pkl'
    data = utils.load_train_data(data_path)

    key_points = data.values()
    key_points = list(key_points)[:10]
    input_length = len(key_points)
    max_length = 19
    hidden_size = 10
    input_size = 2
    output_size = 2

    # Encoder
    EOS = torch.Tensor([0, 0])
    input_tensor = torch.Tensor(key_points)
    input_tensor = torch.cat((input_tensor, EOS.repeat((10, 1, 1))), dim=1)
    encoder = EncoderRNN(input_size, hidden_size, device)
    encoder_hidden = encoder.initHidden(input_length)

    encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)


    decoder = AttnDecoderRNN(hidden_size, output_size, max_length, device)
    decoder_hidden = encoder_hidden
    input = torch.Tensor([-1, -1]) # SOS_token
    input = input.repeat(2, 1)
    decoder(input, decoder_hidden, encoder_output)

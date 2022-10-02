import torch.nn as nn
import torch
import torch.utils.data as Data

class Bidirectional_LSTM(nn.Module):
    def __init__(self, opt):
        super(Bidirectional_LSTM, self).__init__()
        self.input_dim = opt.input_dim
        self.output_dim = opt.output_dim
        self.embedding_dim = opt.embedding_dim
        self.hidden_dim = opt.hidden_dim
        self.n_layers = opt.n_layers
        self.bidirectional = opt.bidirectional
        self.device = opt.device
        self.grid_size_tensor = opt.grid_size_tensor

        self.h_embedding = nn.Embedding(opt.h_grid+1, self.embedding_dim)
        self.w_embedding = nn.Embedding(opt.w_grid+1, self.embedding_dim)

        self.rnn = nn.LSTM(self.embedding_dim * 2 + 1,
                           self.hidden_dim,
                           num_layers=opt.n_layers,
                           bidirectional=self.bidirectional,
                           batch_first=True)
        # self.gru = nn.GRU(input_dim, hidden_dim, batch_first= True)
        self.fc = nn.Linear(self.hidden_dim * 2 if self.bidirectional else self.hidden_dim,
                            self.output_dim)
        self.dropout = nn.Dropout(opt.dropout)

    def forward(self, input):
        if len(input.size()) != 3 :
            input = input.unsqueeze(0)
        grid_embedding_index = torch.div(input[:, :, :-1] + 1, self.grid_size_tensor[None, None, :], rounding_mode='trunc').int()
        h_embedding = self.h_embedding(grid_embedding_index[:, :, 0])
        w_embedding = self.w_embedding(grid_embedding_index[:, :, 1])
        output = torch.cat([h_embedding, w_embedding, input[:, :, 2].unsqueeze(2)], dim = 2)

        output, (hidden, cell) = self.rnn(output)

        output = self.fc(self.dropout(output))
        return output


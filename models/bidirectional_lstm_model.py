import torch.nn as nn
import torch
import torch.utils.data as Data

from data.ochfashion_dataset import MyDataSet

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


from util import util

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__' :
    height = 256
    width = 256
    h_grid = 100
    w_grid = 100

    # model params
    input_dim = 3
    output_dim = 2
    embedding_dim = 3
    h_grid_size = 2 / h_grid
    w_grid_size = 2 / w_grid
    hidden_dim = 5
    n_layers = 3
    bidirectional = True
    dropout = 0.1

    # training params
    batch_size = 10
    learning_rate = 0.005
    n_iters = 100

    data_path = 'dataset/train/pose_label.pkl'
    data_dict = utils.load_train_data(data_path)
    mydata = MyDataSet(data_dict, height, width)

    dataloader = Data.DataLoader(mydata, batch_size, True)

    lstm = Bidirectional_LSTM(input_dim=input_dim,
                      output_dim=output_dim,
                      embedding_dim=embedding_dim,
                      h_grid=h_grid,
                      w_grid=w_grid,
                      hidden_dim=hidden_dim,
                      n_layers=n_layers,
                      bidirectional=bidirectional,
                      dropout=dropout,
                      device=device)

    grid_size_tensor = torch.Tensor([h_grid_size, w_grid_size])
    for src, tgt, mid_point, length in dataloader :
        src, tgt = src.float(), tgt.float()
        lstm(src, grid_size_tensor)


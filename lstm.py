from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.utils.data as Data

import torch.nn.functional as F
import matplotlib.pyplot as plt

import utils
from data.mydata import MyDataSet
from models.encoder import Bidirectional_LSTM
from train_tools import Trainer
from eval_tools import Evaler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





if __name__ == "__main__":
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
    dropout = 0.5

    # training params
    batch_size = 64
    learning_rate = 0.005
    n_epochs = 5

    data_path = 'dataset/train/pose_label.pkl'
    data_dict = utils.load_train_data(data_path)
    mydata = MyDataSet(data_dict, height, width)

    dataloader = Data.DataLoader(mydata, batch_size, True)
    grid_size_tensor = torch.Tensor([h_grid_size, w_grid_size])

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


    trainer = Trainer(model = lstm,
                      grid_size_tensor=grid_size_tensor,
                      device=device)

    trainer.trainIters(n_epochs = n_epochs,
                       print_every=1,
                       plot_every = 1,
                       dataloader=dataloader)

    eval = Evaler(model=lstm,
                  grid_size_tensor=grid_size_tensor,
                  dataloader=dataloader,
                  device=device)
    eval.evaluateRandomly()

    src, tgt = random.choice(dataloader.dataset)

    output_words, attentions = eval.evaluate(src)
    plt.matshow(attentions.numpy())
    plt.savefig("attentions.png")



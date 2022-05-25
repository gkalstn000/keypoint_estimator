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
from models.bidirectional_lstm import Bidirectional_LSTM
from train_tools import Trainer
from eval_tools import Evaler
from options.bidirectional_lstm_options import Bidirectional_LSTM_option

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





if __name__ == "__main__":
    parser = Bidirectional_LSTM_option()
    opt = parser.parse()
    parser.save()

    h_grid_size = 2 / opt.h_grid # (1 - (-1)) / opt.h_grid
    w_grid_size = 2 / opt.w_grid # (1 - (-1)) / opt.w_grid

    data_path = 'dataset/train/pose_label.pkl'
    data_dict = utils.load_train_data(data_path)
    mydata = MyDataSet(data_dict, opt)

    dataloader = Data.DataLoader(mydata, opt.batch_size, True)
    grid_size_tensor = torch.Tensor([h_grid_size, w_grid_size])

    lstm = Bidirectional_LSTM(opt=opt, device=device)

    trainer = Trainer(opt=opt,
                      model = lstm,
                      grid_size_tensor=grid_size_tensor,
                      device=device)

    trainer.trainIters(opt=opt, dataloader=dataloader)

    eval = Evaler(model=lstm,
                  grid_size_tensor=grid_size_tensor,
                  dataloader=dataloader,
                  device=device)
    eval.evaluateRandomly()

    src, tgt = random.choice(dataloader.dataset)

    output_words = eval.evaluate(src)



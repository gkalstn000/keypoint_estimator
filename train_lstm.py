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
from data.mydata import MyDataSet, Make_batch
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
    src_norm_with_unknown, tgt, mid_point, length = Make_batch(data_dict, opt).get_batch()
    mydata = MyDataSet(src_norm_with_unknown, tgt, mid_point, length)

    dataloader = Data.DataLoader(mydata, opt.batch_size, True)
    grid_size_tensor = torch.Tensor([h_grid_size, w_grid_size])

    model = Bidirectional_LSTM(opt=opt, device=device)
    model_optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)

    model, model_optimizer, epoch, loss = utils.load_model(opt, model, model_optimizer)
    opt.epoch = epoch

    trainer = Trainer(opt=opt,
                      model = model,
                      grid_size_tensor=grid_size_tensor,
                      device=device)

    trainer.trainIters(opt=opt,
                       model_optimizer = model_optimizer,
                       dataloader=dataloader)

    eval = Evaler(model=model,
                  grid_size_tensor=grid_size_tensor,
                  dataloader=dataloader,
                  device=device)
    eval.evaluateRandomly()

    src, tgt = random.choice(dataloader.dataset)

    output_words = eval.evaluate(src)



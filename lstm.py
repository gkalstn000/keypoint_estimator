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
from models.encoder import EncoderRNN, DecoderRNN
from models.network import AttnDecoderRNN
from train_tools import Trainer
from eval_tools import Evaler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





if __name__ == "__main__":

    MAX_LENGTH = 19

    height = 256
    width = 256

    input_size = 2
    output_size = 2
    hidden_size = 4
    batch_size = 128
    learning_rate = 0.005
    n_iters = 100

    teacher_forcing_ratio = 0.5

    data_path = 'dataset/train/pose_label.pkl'
    data_dict = utils.load_train_data(data_path)
    mydata = MyDataSet(data_dict, height, width)

    dataloader = Data.DataLoader(mydata, batch_size, True)

    encoder1 = EncoderRNN(input_size, hidden_size, device).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_size, MAX_LENGTH, device, dropout_p=0.1).to(device)
    trainer = Trainer(encoder = encoder1,
                      decoder=attn_decoder1,
                      max_length=MAX_LENGTH,
                      device=device,
                      teacher_forcing_ratio=teacher_forcing_ratio)

    trainer.trainIters(n_iters = n_iters,
                       print_every=1,
                       plot_every = 1,
                       dataloader=dataloader)

    eval = Evaler(encoder=encoder1,
                  decoder=attn_decoder1,
                  max_length=MAX_LENGTH,
                  dataloader=dataloader,
                  device=device)
    eval.evaluateRandomly()

    src, tgt = random.choice(dataloader.dataset)

    output_words, attentions = eval.evaluate(src)
    plt.matshow(attentions.numpy())
    plt.savefig("attentions.png")



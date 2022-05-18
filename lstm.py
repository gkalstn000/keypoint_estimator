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
    '''
    MASK point = [-1, -1]
    SOS point = [0, -1]
    EOS point = [-1, 0]
    '''
    MASK_point = [-1, -1]
    SOS_point = [0, -1]
    EOS_point = [-1, 0]
    MAX_LENGTH = 19

    height = 256
    width = 256

    input_size = 2
    output_size = 2
    hidden_size = 10
    batch_size = 64

    teacher_forcing_ratio = 0.5
    batch_size = 10

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
                      SOS_token=SOS_token,
                      EOS_token=EOS_token,
                      teacher_forcing_ratio=teacher_forcing_ratio)

    trainer.trainIters(n_iters = 75000,
                       pairs = pairs,
                       input_lang= input_lang,
                       output_lang=output_lang,
                       print_every=5000)

    eval = Evaler(encoder=encoder1,
                  decoder=attn_decoder1,
                  pairs=pairs,
                  max_length=MAX_LENGTH,
                  input_lang=input_lang,
                  output_lang=output_lang,
                  device=device,
                  SOS_token=SOS_token,
                  EOS_token=EOS_token)
    eval.evaluateRandomly()

    output_words, attentions = eval.evaluate(sentence="je suis trop froid .")
    plt.matshow(attentions.numpy())


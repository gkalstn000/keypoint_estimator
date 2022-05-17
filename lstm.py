from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


import data.utils as data
from models.encoder import EncoderRNN, DecoderRNN
from models.network import AttnDecoderRNN
import train_tools
import eval_tools


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





if __name__ == "__main__":
    SOS_token = 0
    EOS_token = 1
    MAX_LENGTH = 10

    input_lang, output_lang, pairs = data.prepareData('eng', 'fra', MAX_LENGTH, True)
    print(random.choice(pairs))

    teacher_forcing_ratio = 0.5

    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size, device).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, MAX_LENGTH, device, dropout_p=0.1).to(device)

    train_tools.trainIters(encoder = encoder1,
                           decoder = attn_decoder1,
                           n_iters = 100,
                           pairs = pairs,
                           input_lang= input_lang,
                           output_lang=output_lang,
                           print_every=5000,
                           EOS_token=EOS_token,
                           SOS_token=SOS_token,
                           device=device,
                           max_length=MAX_LENGTH,
                           teacher_forcing_ratio=teacher_forcing_ratio)
    eval_tools.evaluateRandomly(encoder=encoder1,
                                decoder=attn_decoder1,
                                pairs=pairs,
                                max_length=MAX_LENGTH,
                                input_lang=input_lang,
                                output_lang=output_lang,
                                device=device,
                                SOS_token=SOS_token,
                                EOS_token=EOS_token)

    output_words, attentions = eval_tools.evaluate(encoder=encoder1,
                                                   decoder=attn_decoder1,
                                                   sentence="je suis trop froid .",
                                                   max_length=MAX_LENGTH,
                                                   input_lang=input_lang,
                                                   output_lang=output_lang,
                                                   device=device,
                                                   SOS_token=SOS_token,
                                                   EOS_token=EOS_token)
    plt.matshow(attentions.numpy())


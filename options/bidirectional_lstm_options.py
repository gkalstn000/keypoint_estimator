from __future__ import division, print_function

import torch
import argparse
import os
import util.io as io
from options.base_options import Base_option

class Bidirectional_LSTM_option(Base_option):
    def initialize(self):
        super(Bidirectional_LSTM_option, self).initialize()
        parser = self.parser
        # model params
        parser.add_argument('--input_dim', type=int, default=3, help='input dimension')
        parser.add_argument('--output_dim', type=int, default=2, help='output dimension')
        parser.add_argument('--embedding_dim', type=int, default=3, help='grid embedding dimension')
        parser.add_argument('--hidden_dim', type=int, default=5, help='hidden state dimension')
        parser.add_argument('--n_layers', type=int, default=3, help='number of bidirectional lstm layers')
        parser.add_argument('--bidirectional', action='store_true', help='bidirectional condition')
        parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')

if __name__ == "__main__":
    parser = Bidirectional_LSTM_option()
    opt = parser.parse()
    parser.save()
    parser.load('train_opt.json')
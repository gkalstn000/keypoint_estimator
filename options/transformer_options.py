from __future__ import division, print_function

import torch
import argparse
import os
import util.io as io
from options.base_options import Base_option


class Transformer(Base_option):
    def initialize(self):
        super(Transformer, self).initialize()
        parser = self.parser
        # model params
        parser.add_argument('--input_dim', type=int, default=3, help='input dimension')
        parser.add_argument('--output_dim', type=int, default=2, help='output dimension')
        parser.add_argument('--embedding_dim', type=int, default=5, help='grid embedding dimension')
        parser.add_argument('--n_layers', type=int, default=5, help='number of bidirectional lstm layers')
        parser.add_argument('--d_k', type=int, default=4, help='number of bidirectional lstm layers')
        parser.add_argument('--d_v', type=int, default=4, help='number of bidirectional lstm layers')
        parser.add_argument('--n_heads', type=int, default=4, help='number of bidirectional lstm layers')
        parser.add_argument('--d_ff', type=int, default=128, help='number of bidirectional lstm layers')

from __future__ import division, print_function

import torch
import argparse
import os
import util.io as io

def opt_to_str(opt):
    return '\n'.join(['%s: %s' % (str(k), str(v)) for k, v in sorted(vars(opt).items())])


class Transformer_option(object):

    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False
        self.opt = None

    def initialize(self):
        parser = self.parser
        # path params
        parser.add_argument('--id', type=str, default='default', help='experiment ID. the experiment dir will be set as "./checkpoint/id/"')
        parser.add_argument('--model_name', type=str, default='model_param_latest', help='load model params with file name')

        # input params
        parser.add_argument('--height', type=int, default=256, help='height of image')
        parser.add_argument('--width', type=int, default=256, help='width of image')
        parser.add_argument('--h_grid', type=int, default=100, help='number of height embedding')
        parser.add_argument('--w_grid', type=int, default=100, help='number of height embedding')
        # model params
        parser.add_argument('--input_dim', type=int, default=3, help='input dimension')
        parser.add_argument('--output_dim', type=int, default=2, help='output dimension')
        parser.add_argument('--embedding_dim', type=int, default=3, help='grid embedding dimension')
        parser.add_argument('--n_layers', type=int, default=3, help='number of bidirectional lstm layers')
        parser.add_argument('--d_k', type=int, default=3, help='number of bidirectional lstm layers')
        parser.add_argument('--d_v', type=int, default=3, help='number of bidirectional lstm layers')
        parser.add_argument('--n_heads', type=int, default=5, help='number of bidirectional lstm layers')
        parser.add_argument('--d_ff', type=int, default=128, help='number of bidirectional lstm layers')

        # training params
        parser.add_argument('--mode', type=str, default='train', help='set mode [train/test]')
        parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
        parser.add_argument('--learning_rate', type=float, default=0.005, help='learning rate')
        parser.add_argument('--n_epochs', type=int, default=200, help='number of epoch')
        parser.add_argument('--continue_train', action='store_true', help='continue_training')

        parser.add_argument('--save_epoch', type=int, default=200, help='model save epoch step')
        parser.add_argument('--print_every', type=int, default=200, help='print loss epoch step')
        parser.add_argument('--plot_every', type=int, default=200, help='plot loss epoch step')

        self.initialized = True

    def parse(self, display=True):
        '''
        Parse option from terminal command string. If ord_str is given, parse option from it instead.
        '''

        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()

        # display options
        if display:
            print('------------ Options -------------')
            for k, v in sorted(vars(self.opt).items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')
        return self.opt

    def save(self, fn=None):
        if self.opt is None:
            raise Exception("parse options before saving!")
        if fn is None:
            expr_dir = os.path.join('checkpoints', self.opt.id)
            io.mkdir_if_missing(expr_dir)
            if self.opt.mode == 'train':
                fn = os.path.join(expr_dir, 'train_opt.json')
            else:
                fn = os.path.join(expr_dir, 'test_opt.json')
        io.save_json(vars(self.opt), fn)

    def load(self, fn):
        args = io.load_json(fn)
        return argparse.Namespace(**args)
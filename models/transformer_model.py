import re
import math
import torch
import numpy as np
import torch.nn as nn
from .networks.transformer_network import *

def fc_layer(size_in, size_out, keep_prob=0.9):
    linear = nn.Linear(size_in, size_out)

    layer = nn.Sequential(
        linear,
        # nn.BatchNorm1d(18),
        nn.Dropout(p=1 - keep_prob),
        nn.ReLU()
    )

    return layer

class Transformer(nn.Module):
    def __init__(self, opt):
        super(Transformer, self).__init__()
        self.h_grid = opt.h_grid
        self.w_grid = opt.w_grid
        self.input_dim = opt.input_dim
        self.output_dim = opt.output_dim
        self.embedding_dim = opt.embedding_dim
        self.n_layers = opt.n_layers
        self.d_k = opt.d_k
        self.d_v = opt.d_v
        self.n_heads = opt.n_heads
        self.d_ff = opt.d_ff
        self.grid_size_tensor = opt.grid_size_tensor
        self.sigmoid = nn.Sigmoid()

        self.embedding = Embedding(h_grid=self.h_grid, w_grid=self.w_grid, embedding_dim=self.embedding_dim)

        d_model = self.embedding_dim * 2 + 1
        self.layers = nn.ModuleList([EncoderLayer(d_model = d_model, d_k = self.d_k, d_v = self.d_v, n_heads = self.n_heads, d_ff = self.d_ff) for _ in range(self.n_layers)])

        self.mask_kernel = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        # fc2 is shared with embedding layer
        # self.MLM_Regressor = nn.Linear(d_model, self.output_dim)
        self.MLM_Regressor = nn.ModuleList()
        input_ = d_model
        for output in [self.d_ff, 32]:
            self.MLM_Regressor.append(fc_layer(input_, output))
            input_ = output
        layer = nn.Sequential(
            nn.Linear(input_, self.output_dim),
            # nn.BatchNorm1d(18),
        )
        self.MLM_Regressor.append(layer)

        self.Classifier = nn.ModuleList()
        input_ = d_model # embedding*2+1
        for output in [self.d_ff, 32]:
            self.Classifier.append(fc_layer(input_, output))
            input_ = output
        layer = nn.Sequential(
            nn.Linear(input_, 1),
            nn.Sigmoid()
        )
        self.Classifier.append(layer)
    def forward(self, x):
        if len(x.size()) != 3 :
            x = x.unsqueeze(0)
        output = self.embedding(x, self.grid_size_tensor) # [bach_size, seq_len, d_model]

        for layer in self.layers:
            # output: [batch_size, max_len, d_model]
            output = layer(output)

        output = self.activ2(self.mask_kernel(output)) # [batch_size, max_pred, d_model]

        keypoint_logits = output
        for linear in self.MLM_Regressor :
            keypoint_logits = linear(keypoint_logits)

        occlusion_logits = output
        for linear in self.Classifier :
            occlusion_logits = linear(occlusion_logits)

        return keypoint_logits, occlusion_logits

if __name__ == '__main__' :
    parser = Transformer()
    opt = parser.parse()
    parser.save()
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    h_grid_size = 2 / opt.h_grid # (1 - (-1)) / opt.h_grid
    w_grid_size = 2 / opt.w_grid # (1 - (-1)) / opt.w_grid

    data_path = 'dataset/train/pose_label.pkl'
    print('Make Batch Dataset', end = '...')
    data_dict = utils.load_train_data(data_path)
    src_norm_with_unknown, tgt, mid_point, length = Make_batch(data_dict, opt).get_batch()
    train_index, test_index = split_data(src_norm_with_unknown, tgt, mid_point, length)

    mydata = MyDataSet(src_norm_with_unknown[train_index, :, :], tgt[train_index, :, :], mid_point, length)
    print('Done!!', end = '...')
    dataloader = Data.DataLoader(mydata, opt.batch_size, True)
    grid_size_tensor = torch.Tensor([h_grid_size, w_grid_size])
    model = Transformer(opt, grid_size_tensor)

    for src_norm_with_unknown, tgt, mid_point, length in dataloader :
        src_norm_with_unknown, tgt, mid_point, length = \
            src_norm_with_unknown.float(), tgt.float(), mid_point.float(), length.float()
        out = model(src_norm_with_unknown)
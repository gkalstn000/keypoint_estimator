from __future__ import unicode_literals, print_function, division

import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.data as Data

import util.io
import utils
from data.mydata import MyDataSet, Make_batch, split_data
from models.bidirectional_lstm import Bidirectional_LSTM
from tools.train_tools import Trainer
from tools.eval_tools import Evaler
from options.bidirectional_lstm_options import Bidirectional_LSTM_option

import os
import scores.point_scores as scores

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



if __name__ == "__main__":
    parser = Bidirectional_LSTM_option()
    opt = parser.parse()
    opt.device = device

    h_grid_size = 2 / opt.h_grid # (1 - (-1)) / opt.h_grid
    w_grid_size = 2 / opt.w_grid # (1 - (-1)) / opt.w_grid

    data_path = 'dataset/train/pose_label.pkl'
    print('Make Batch Dataset', end = '...')
    data_dict = utils.load_train_data(data_path)
    src_norm_with_unknown, tgt, mid_point, length = Make_batch(data_dict, opt).get_batch()

    train_index, test_index = split_data(src_norm_with_unknown, tgt, mid_point, length)

    mydata = MyDataSet(src_norm_with_unknown[test_index, :, :], tgt[test_index, :, :], mid_point, length)
    print('Done!!')

    dataloader = Data.DataLoader(mydata, opt.batch_size, True)
    grid_size_tensor = torch.Tensor([h_grid_size, w_grid_size])

    model = Bidirectional_LSTM(opt=opt)
    model_optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    scheduler = ReduceLROnPlateau(model_optimizer, 'min', verbose=True)

    model, _, _, _, _ = utils.load_model(opt, model, model_optimizer, scheduler)

    eval = Evaler(model=model,
                  grid_size_tensor=grid_size_tensor,
                  dataloader=dataloader,
                  device=device)

    src, tgt, pred = eval.evaluateRandomly(score=scores.score)

    plot_save_path = f'checkpoints/{opt.id}/figure'
    util.io.mkdir_if_missing((plot_save_path))
    for i in range(src.shape[0]) :
        utils.plot_key_points(src[i], tgt[i], pred[i], os.path.join(plot_save_path, f'compare_figure_{i}.png'))

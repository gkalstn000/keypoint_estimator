from __future__ import unicode_literals, print_function, division

import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.data as Data

import utils
from data.mydata import MyDataSet, Make_batch, split_data
from models.bidirectional_lstm_model import Bidirectional_LSTM
from tools.train_tools import Trainer
from tools.eval_tools import Evaler
from options.base_options import Base_option
from models import create_model
from options import create_option

if __name__ == "__main__":
    base_opt = Base_option().parse()
    parser = create_option(base_opt)
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
    opt.grid_size_tensor = grid_size_tensor

    model = create_model(opt)
    model_optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    scheduler = ReduceLROnPlateau(model_optimizer, 'min', verbose=True)

    model, model_optimizer, scheduler, epoch, loss = utils.load_model(opt, model, model_optimizer, scheduler)
    opt.epoch = epoch
    opt.loss = loss
    # opt.epoch = 1
    # opt.loss = 0
    trainer = Trainer(opt=opt,
                      model = model)

    trainer.trainIters(opt=opt,
                       model_optimizer = model_optimizer,
                       scheduler = scheduler,
                       dataloader=dataloader)




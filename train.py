# =========== Import pytorch libs ===========
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.data as Data
# =========== Import Option modules ===========
from options.train_options import TrainOptions
# =========== Import Dataset modules ===========
from data.tmp_dataset import MyDataSet, Make_batch
import data
# =========== Import Models ===========
from models import create_model
# =========== Import util modules ===========
from util import util
from tools.train_tools import Trainer
# =========== Import etc... ===========
import sys

if __name__ == "__main__":
    # parse options
    opt = TrainOptions().parse()
    # print options to help debugging
    print(' '.join(sys.argv))

    dataloader = data.create_dataloader(opt)

    # cuda는 model안에서 바꿔주는걸로
    for i, data_i in enumerate(dataloader):
        print(data_i)


    #
    # h_grid_size = 2 / opt.h_grid # (1 - (-1)) / opt.h_grid
    # w_grid_size = 2 / opt.w_grid # (1 - (-1)) / opt.w_grid
    #
    # # data_path = 'dataset/train/pose_label.pkl'
    # print('Make Batch Dataset', end = '...')
    # train_df = utils.load_train_data('dataset/train_annotation.csv')
    # src, tgt_with_occlusion, mid_point, length = Make_batch(train_df, opt).get_batch()
    # train_data = MyDataSet(src, tgt_with_occlusion, mid_point, length)
    #
    # valid_df = utils.load_train_data('dataset/valid_annotation.csv')
    # src, tgt_with_occlusion, mid_point, length = Make_batch(valid_df, opt).get_batch()
    # valid_data = MyDataSet(src, tgt_with_occlusion, mid_point, length)
    #
    # print('Done!!', end = '...')
    # train_dl = Data.DataLoader(train_data, opt.batch_size, True)
    # valid_dl = Data.DataLoader(valid_data, opt.batch_size, True)
    #
    # grid_size_tensor = torch.Tensor([h_grid_size, w_grid_size])
    # opt.grid_size_tensor = grid_size_tensor.to(opt.device)
    #
    # model = create_model(opt)
    # model_optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    # scheduler = ReduceLROnPlateau(model_optimizer, 'min', verbose=True)
    #
    # model, model_optimizer, scheduler, epoch, loss = utils.load_model(opt, model, model_optimizer, scheduler)
    # opt.epoch = epoch
    # opt.loss = loss
    # # opt.epoch = 1
    # # opt.loss = 0
    # trainer = Trainer(opt=opt,
    #                   model=model)
    #
    # trainer.trainIters(opt=opt,
    #                    model_optimizer=model_optimizer,
    #                    scheduler=scheduler,
    #                    train_dl=train_dl,
    #                    valid_dl=valid_dl)




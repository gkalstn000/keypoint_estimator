from __future__ import unicode_literals, print_function, division

import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.data as Data

import util.io
from util import util
from data.ochfashion_dataset import MyDataSet, Make_batch
from tools.eval_tools import Evaler
from options.transformer_options import Transformer as transformer_option

from models import create_model
import os
import scores.scores as scores
from tqdm import trange

if __name__ == "__main__":
    # base_opt = Base_option().parse()
    # parser = create_option(base_opt)
    # opt = parser.parse()
    # assert opt.mode == 'test', 'mode is not test'

    parser = transformer_option()
    opt = parser.parse()
    parser.save()
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # parser.save()

    h_grid_size = 2 / opt.h_grid # (1 - (-1)) / opt.h_grid
    w_grid_size = 2 / opt.w_grid # (1 - (-1)) / opt.w_grid

    data_path = 'dataset/test_annotation.csv'
    print('Make Batch Dataset', end = '...')
    test_df = utils.load_train_data(data_path)
    src, tgt_with_occlusion, mid_point, length = Make_batch(test_df, opt).get_batch()
    mydata = MyDataSet(src, tgt_with_occlusion, mid_point, length)
    print('Done!!')

    dataloader = Data.DataLoader(mydata, opt.batch_size, True)
    grid_size_tensor = torch.Tensor([h_grid_size, w_grid_size])

    opt.grid_size_tensor = grid_size_tensor.to(opt.device)
    model = create_model(opt)

    model_optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    scheduler = ReduceLROnPlateau(model_optimizer, 'min', verbose=True)

    model, _, _, _, _ = utils.load_model(opt, model, model_optimizer, scheduler)
    model = model.to(opt.device)
    eval = Evaler(opt=opt,
                  model=model,
                  dataloader=dataloader)


    src, tgt, pred, occ_true, occ_pred, keypoint_df, occlusion_df = eval.evaluate_score(score=scores.score)
    plot_save_path = f'checkpoints/{opt.model}/{opt.id}/figure'

    keypoint_df.to_csv(f'checkpoints/{opt.model}/{opt.id}/keypoint_score.csv')
    occlusion_df.to_csv(f'checkpoints/{opt.model}/{opt.id}/occlusion_score.csv')

    util.io.mkdir_if_missing((plot_save_path))
    if opt.save_eval_image :
        for i in trange(src.shape[0]) :
            utils.plot_key_points(src[i], tgt[i], pred[i], occ_true[i], occ_pred[i], os.path.join(plot_save_path, f'compare_figure_{i}.png'))


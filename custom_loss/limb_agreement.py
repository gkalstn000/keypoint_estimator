import torch
import torch.nn as nn
from utils import single_plot_key_points

def cal_limb_agreement(output, target) :
    output_directions = get_direction_vector(output)
    target_directions = get_direction_vector(target)
    agreements = (output_directions * target_directions).sum(-1) / (torch.norm(output_directions, dim = -1) * (torch.norm(target_directions, dim = -1)))
    n_loglikelihood = -torch.log(torch.abs(agreements))
    return n_loglikelihood.nansum(-1).nanmean()
def get_direction_vector(points) :
    vectors = [[1, 0], [1, 2], [1, 8], [1, 11], [1, 5], [2, 3], [3, 4], [8, 9], [9, 10], [11, 12], [12, 13], [5, 6], [6, 7], [0, 14], [14, 16], [0, 15], [15, 17]]
    return get_relationship(points, vectors)


def get_relationship(points, vector) :
    buffer = []
    for src, dest in vector :
        buffer.append(points[:, dest] - points[:, src])
    return torch.stack(buffer, dim=1)




import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.data as Data

import utils
from data.mydata import MyDataSet, Make_batch, split_data
from tools.eval_tools import Evaler
from options.base_options import Base_option
from models import create_model
from options import create_option
import scores.scores as scores
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

    mydata = MyDataSet(src_norm_with_unknown[test_index, :, :], tgt[test_index, :, :], mid_point, length)
    print('Done!!')

    dataloader = Data.DataLoader(mydata, opt.batch_size, True)
    grid_size_tensor = torch.Tensor([h_grid_size, w_grid_size])
    opt.grid_size_tensor = grid_size_tensor
    model = create_model(opt)
    model_optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    scheduler = ReduceLROnPlateau(model_optimizer, 'min', verbose=True)

    model, _, _, _, _ = utils.load_model(opt, model, model_optimizer, scheduler)

    eval = Evaler(opt=opt,
                  model=model,
                  dataloader=dataloader)

    src, tgt, pred, score_df = eval.evaluate_score(score=scores.score)

    target = torch.Tensor(tgt)
    output = torch.Tensor(pred)

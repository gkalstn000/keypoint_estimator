import torch
import torch.nn as nn
from utils import single_plot_key_points

def cal_pose_loss(output, target) :
    output_energy = get_energy(output)
    target_energy = get_energy(target)

    return torch.abs((output_energy - target_energy)).mean()
def get_energy(points) :
    B_vector = [[1, 0], [1, 2], [1, 8], [1, 11], [1, 5]] # body
    RA_vector = [[2, 3], [3, 4]] # right arm
    LA_vector = [[5, 6], [6, 7]] # left arm
    RL_vector = [[8, 9], [9, 10]] # right leg
    LL_vector = [[11, 12], [12, 13]] # left leg
    RF_vector = [[0, 14], [14, 16]] # right face
    LF_vector = [[0, 15], [15, 17]] # left face

    body_matrix = get_relationship(points, B_vector)
    RA_matrix = get_relationship(points, RA_vector)
    LA_matrix = get_relationship(points, LA_vector)
    RL_matrix = get_relationship(points, RL_vector)
    LL_matrix = get_relationship(points, LL_vector)
    RF_matrix = get_relationship(points, RF_vector)
    LF_matrix = get_relationship(points, LF_vector)

    RAB = body_matrix @ RA_matrix
    LAB = body_matrix @ LA_matrix
    RLB = body_matrix @ RL_matrix
    LLB = body_matrix @ LL_matrix
    RFB = body_matrix @ RF_matrix
    LFB = body_matrix @ LF_matrix

    RA = limb_with_body(RAB)
    LA = limb_with_body(LAB)
    RL = limb_with_body(RLB)
    LL = limb_with_body(LLB)
    RF = limb_with_body(RFB)
    LF = limb_with_body(LFB)

    # return torch.log(torch.abs(RA+LA+RL+LL+RF+LF))
    return torch.abs(RA+LA+RL+LL+RF+LF)


def limb_with_body(matrix) :
    return torch.det(matrix.transpose(2, 1) @ matrix)

def get_relationship(points, vector) :
    buffer = []
    for src, dest in vector :
        buffer.append(points[:, dest] - points[:, src])
    return torch.stack(buffer, dim=1)




import torch
import torch.utils.data as Data

import utils
from data.mydata import MyDataSet, Make_batch, split_data
from options.base_options import Base_option
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

    mydata = MyDataSet(src_norm_with_unknown[test_index, :, :], tgt[test_index, :, :], mid_point, length)
    print('Done!!')

    dataloader = Data.DataLoader(mydata, opt.batch_size, True)

    for src, tgt, mid_point, length in dataloader:
        losses = get_energy(tgt)
        break
    pose_loss_dict = {k: v for k, v in zip(losses, tgt)}
    pose_loss_dict = dict(sorted(pose_loss_dict.items(), key=lambda item: item[0]))
    for loss, pose in pose_loss_dict.items() :
        single_plot_key_points(pose, loss)

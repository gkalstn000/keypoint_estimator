import numpy as np
from numpy import random
import json

np.random.seed(seed=100)

import torch.utils.data as Data
import torch

def set_random(seed = 1004) :
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

class MyDataSet(Data.Dataset) :
    def __init__(self, src, tgt, mid_point, length):
        self.src = src
        self.tgt = tgt
        self.mid_point = mid_point
        self.length = length

    def __len__(self):
        return self.tgt.shape[0]
    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx], self.mid_point, self.length

class Make_batch:
    def __init__(self, data_df, opt):
        self.data_array = df_to_array(data_df)
        self.height = opt.height
        self.width = opt.width

        self.alpha = opt.alpha
        self.beta = opt.beta
        self.opt = opt



    def get_batch(self):
        self.key_points = self.data_array[:, :, :2]


        R, R_inv, T = self.affine_matrix_batch()



        # occlusion 좌표 index
        occlusion_index = self.data_array[:, :,2]


        scaling = self.key_points @ R_inv
        moving = R_inv @ np.expand_dims(T, axis=2)
        tgt = scaling - moving.transpose(0, 2, 1)  # (B, L, 2)
        tgt[occlusion_index] = np.array([self.height + 1, self.width + 1])
        # Normalization을 위한 values
        max_point = np.array([[self.height, self.width]])
        min_point = np.array([[0, 0]])
        mid_point = (max_point + min_point) / 2
        length = max_point - min_point

        cond1 = tgt > max_point.squeeze()
        cond2 = tgt < min_point.squeeze()

        cond1_logical = np.logical_or(cond1[:, :, 0], cond1[:, :, 1])
        cond2_logical = np.logical_or(cond2[:, :, 0], cond2[:, :, 1])

        unknown_index = np.logical_or(cond1_logical, cond2_logical)
        # 없애면 1, 안 없애면 0
        unknown_token = np.expand_dims(unknown_index, axis = 2)

        src = tgt.copy()
        src[unknown_index] = np.array([self.height + 1, self.width + 1])
        src[occlusion_index] = np.array([self.height + 1, self.width + 1])
        src_norm = (src-mid_point) * 2 / length
        tgt_with_occlusion = np.concatenate((tgt, np.expand_dims(occlusion_index, 2)), axis = 2)
        return src_norm, tgt_with_occlusion, mid_point, length


    def affine_matrix_batch(self):

        Rs, R_invs, Ts = [], [], []
        for _ in range(self.key_points.shape[0]):
            R, R_inv, T = self.affine_matrix()
            Rs.append(R)
            R_invs.append(R_inv)
            Ts.append(T)

        R = np.stack(Rs, axis=0)
        R_inv = np.stack(R_invs, axis=0)
        T = np.stack(Ts, axis=0)

        return R, R_inv, T

    def affine_matrix(self):
        h_blank = self.height / 10
        w_blank = self.width / 10
        while True:
            h_scale, w_scale = random.beta(self.alpha, self.beta, size=2)
            h_move = random.uniform(0, (1 - h_scale) * self.height)
            w_move = random.uniform(0, (1 - w_scale) * self.width)

            R = np.array([[h_scale, 0],
                          [0, w_scale]])
            T = np.array([h_move, w_move])

            R_inv = np.linalg.inv(R)

            x_max = np.array([self.height, self.width])
            max_transform = R @ x_max + T

            if max_transform[0] + h_blank < self.height and max_transform[1] + w_blank < self.width:
                break

        return R, R_inv, T



def denormalization(points, mid_point, length) :
    return (points * length) / 2 + mid_point

def split_data(src, tgt, mid_point, length) :
    set_random()
    length = src.shape[0]
    test_ratio = 0.3

    total_index = np.arange(length)
    np.random.shuffle(total_index)

    train_index = total_index[int(length*test_ratio):]
    test_index = total_index[:int(length*test_ratio)]

    return train_index, test_index


from util import util

if __name__ == '__main__' :
    data_path = 'dataset/train/pose_label.pkl'
    data_dict = utils.load_train_data(data_path)

    height = 256
    width = 256

    mydata = MyDataSet(data_dict, height, width)
    batch_size = 10
    dataloader = Data.DataLoader(mydata, batch_size, True)

    # for src, tgt, mid_point, length in dataloader :
    #     print(src)
    #     print(tgt)
    #     break

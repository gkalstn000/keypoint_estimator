import torch.utils.data as Data
from numpy import random
import numpy as np
import matplotlib.pyplot as plt

import torch.utils.data as Data
import torch


class MyDataSet(Data.Dataset):
    def __init__(self, data_dict, opt):
        self.file_name = list(data_dict.keys())
        self.key_points = np.array(list(data_dict.values()))
        self.height = opt.height
        self.width = opt.width


    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, idx):
        R, R_inv, T = self.affine_matrix_batch()

        scaling = self.key_points @ R_inv
        moving = R_inv @ np.expand_dims(T, axis=2)
        tgt = scaling - moving.transpose(0, 2, 1)  # (B, L, 2)

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
        src_norm = (src-mid_point) * 2 / length
        src_norm_with_unknown = np.concatenate((src_norm, unknown_token), axis = 2)
        return  src_norm_with_unknown[idx], tgt[idx], mid_point, length


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

    def affine_matrix(self, alpha=32, beta=20):
        h_blank = self.height / 10
        w_blank = self.width / 10
        while True:
            h_scale, w_scale = random.beta(alpha, beta, size=2)
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

def plot_key_points(src, tgt) :
    skeleton_tree = [[16, 14], [14, 0], [15, 0], [17, 15], [0, 1],
                     [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7],
                     [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13]]

    if type(src) != np.ndarray:
        src.numpy()
        tgt.numpy()

    invalid = np.array([-1.0, -1.0])
    # tgt drawing
    for p1, p2 in skeleton_tree:
        h1, w1 = tgt[p1]
        h2, w2 = tgt[p2]
        plt.plot([h1, h2], [w1, w2], color='red')
    # src drawing
    for p1, p2 in skeleton_tree:
        if (src[p1] == invalid).sum() > 0 or (src[p2] == invalid).sum() > 0: continue
        h1, w1 = src[p1]
        h2, w2 = src[p2]
        plt.plot([h1, h2], [w1, w2], color='green')

    plt.show()

def denormalization(points, mid_point, length) :
    return (points * length) / 2 + mid_point

import utils
if __name__ == '__main__' :
    data_path = 'dataset/train/pose_label.pkl'
    data_dict = utils.load_train_data(data_path)

    height = 256
    width = 256

    mydata = MyDataSet(data_dict, height, width)
    batch_size = 10
    dataloader = Data.DataLoader(mydata, batch_size, True)

    for src, tgt, mid_point, length in dataloader :
        print(src)
        print(tgt)
        break

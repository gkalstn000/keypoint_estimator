import torch.utils.data as Data
from numpy import random
import numpy as np
import matplotlib.pyplot as plt

import torch.utils.data as Data
import torch


class MyDataSet(Data.Dataset):
    def __init__(self, data_dict, height, width):
        self.file_name = list(data_dict.keys())
        self.key_points = np.array(list(data_dict.values()))
        self.height = height
        self.width = width


    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, idx):
        R, R_inv, T = self.affine_matrix(self.height, self.width)

        tgt = self.key_points @ R_inv - R_inv @ T # (B, L, 2)

        cond1 = tgt > np.array([self.height, self.width])
        cond2 = tgt < np.array([0, 0])

        cond1_logical = np.logical_or(cond1[:, :, 0], cond1[:, :, 1])
        cond2_logical = np.logical_or(cond2[:, :, 0], cond2[:, :, 1])

        erase_idx = np.logical_or(cond1_logical, cond2_logical)

        src = tgt.copy()
        src[erase_idx] = np.array([-1, -1])
        return src[idx], tgt[idx]



    def affine_matrix(self, height, width, alpha=32, beta=20):
        h_blank = height / 10
        w_blank = width / 10
        while True:
            h_scale, w_scale = random.beta(alpha, beta, size=2)
            h_move = random.uniform(0, (1 - h_scale) * height)
            w_move = random.uniform(0, (1 - w_scale) * width)

            R = np.array([[h_scale, 0],
                          [0, w_scale]])
            T = np.array([h_move, w_move])

            R_inv = np.linalg.inv(R)

            x_max = np.array([height, width])
            max_transform = R @ x_max + T

            if max_transform[0] + h_blank < height and max_transform[1] + w_blank < width:
                break

        return R, R_inv, T

def plot_key_points(src, tgt) :
    skeleton_tree = [[16, 14], [14, 0], [15, 0], [17, 15], [0, 1],
                     [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7],
                     [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13]]

    invalid = torch.Tensor([-1.0, -1.0])
    # tgt drawing
    for p1, p2 in skeleton_tree :
        h1, w1 = tgt[p1]
        h2, w2 = tgt[p2]
        plt.plot([h1, h2], [w1, w2], color = 'red')
    # src drawing
    for p1, p2 in skeleton_tree :
        if (src[p1] == invalid).sum() > 0 or (src[p2] == invalid).sum() > 0 : continue
        h1, w1 = src[p1]
        h2, w2 = src[p2]
        plt.plot([h1, h2], [w1, w2], color = 'green')


    plt.show()

import utils
if __name__ == '__main__' :
    data_path = 'dataset/train/pose_label.pkl'
    data_dict = utils.load_train_data(data_path)

    height = 256
    width = 256

    mydata = MyDataSet(data_dict, height, width)
    batch_size = 10
    dataloader = Data.DataLoader(mydata, batch_size, True)

    for src, tgt in dataloader :
        print(src)
        print(tgt)
        break

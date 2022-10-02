import torch.utils.data as data
import numpy as np
from numpy import random
import json
from tqdm import tqdm, trange

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        pass

def df_to_array(df):
    data_list = []
    for index, (keypoints_y, keypoints_x, label) in tqdm(df.iterrows(), desc='Make DataFrame to numpy array'):
        keypoints_y = json.loads(keypoints_y)
        keypoints_x = json.loads(keypoints_x)
        label = json.loads(label)
        data_list.append([[h, w, l] for h, w, l in zip(keypoints_y, keypoints_x, label)])
    return np.array(data_list)



# =========== Affine Methods ===========
def get_affine_params(height, width, alpha, beta):
    h_scale, w_scale = random.beta(alpha, beta, size=2)
    h_move = random.uniform(0, (1 - h_scale) * (height - 1))
    w_move = random.uniform(0, (1 - w_scale) * (width - 1))
    R = np.array([[h_scale, 0],
                  [0, w_scale]])
    T = np.array([h_move, w_move])
    R_inv = np.linalg.inv(R)

    return R, R_inv, T

def make_affine_params_batch(height, width, alpha, beta, length):
    Rs, R_invs, Ts = [], [], []
    for _ in trange(length, desc='Generate Affine Parameters'):
        R, R_inv, T = get_affine_params(height, width, alpha, beta)
        Rs.append(R)
        R_invs.append(R_inv)
        Ts.append(T)

    R = np.stack(Rs, axis=0) # (length, 2, 2)
    R_inv = np.stack(R_invs, axis=0) # (length, 2, 2)
    T = np.stack(Ts, axis=0)

    return R, R_inv, np.expand_dims(T, 2)
# =========== Affine Methods ===========

def denormalization(points, mid_point, length) :
    return (points * length) / 2 + mid_point


if __name__ == "__main__":
    height, width, alpha, beta, length = 256, 176, 32, 20, 100
    make_affine_params_batch(height, width, alpha, beta, length)
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random
import json

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
    for index, (keypoints_y, keypoints_x, label) in df.iterrows():
        keypoints_y = json.loads(keypoints_y)
        keypoints_x = json.loads(keypoints_x)
        label = json.loads(label)
        data_list.append([[h, w, l] for h, w, l in zip(keypoints_y, keypoints_x, label)])
    return np.array(data_list)
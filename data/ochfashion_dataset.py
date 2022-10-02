import os
import numpy as np
from data.base_dataset import BaseDataset, df_to_array, make_affine_params_batch
from util import util

class OCHFashionDataset(BaseDataset) :
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--affine_alpha', type=float, default=32, help='learning rate')
        parser.add_argument('--affine_beta', type=float, default=20, help='learning rate')
        return parser

    def initialize(self, opt):
        self.keypoint, self.occlusion_label = self.get_data(opt)
        size = len(self.keypoint)
        self.dataset_size = size
        # affine params
        self.max_height = opt.max_height
        self.max_wigth = opt.max_width
        self.affine_alpha = opt.affine_alpha
        self.affine_beta = opt.affine_beta
    def get_data(self, opt):
        root = os.path.join(opt.dataroot, opt.dataset_mode)
        phase = opt.phase
        df = util.load_df(os.path.join(root, f'{phase}_annotation.csv'))
        data = df_to_array(df)
        keypoint, occlusion_label = data[:, :, :2], data[:, :, 2]
        return keypoint, occlusion_label

    def __getitem__(self, index):
        R, R_inv, T = make_affine_params_batch(self.max_height, self.max_wigth, self.affine_alpha, self.affine_beta, self.dataset_size)
        # Inverse Affine transformation으로, keypoint -> target_keypoint generate
        target_keypoint = self.keypoint @ R_inv - (R_inv @ T).transpose(0, 2, 1)

        max_point = np.array([[self.max_height - 1, self.max_wigth - 1]])
        min_point = np.array([[0, 0]])
        cond1 = target_keypoint > max_point.squeeze()
        cond2 = target_keypoint < min_point.squeeze()


        return 0

    def __len__(self):
        return self.dataset_size
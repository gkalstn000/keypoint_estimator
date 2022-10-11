import os
import numpy as np
from data.base_dataset import BaseDataset, df_to_array, make_affine_params_batch, get_affine_params
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
        keypoint = self.keypoint[index]
        occlusion_label = self.occlusion_label[index]
        R, R_inv, T = get_affine_params(self.max_height, self.max_wigth, self.affine_alpha, self.affine_beta)
        # Inverse Affine transformation으로, keypoint -> target_keypoint generate
        target_keypoint = keypoint @ R_inv - R_inv @ T
        target_keypoint[keypoint == -1] = np.nan

        max_point = np.array([self.max_height - 1, self.max_wigth - 1])
        min_point = np.array([0, 0])
        # Invalid Source keypoint condition
        over_cond = (target_keypoint > max_point).any(axis = -1)
        under_cond = (target_keypoint < min_point).any(axis = -1)
        croppd_cond = over_cond | under_cond
        # generate source keypoint from target keypoint
        source_keypoint = target_keypoint.copy()
        source_keypoint[croppd_cond] = np.nan
        # Normalize Source keypoint [0, 1] <--Min-Max norm
        source_keypoint /= max_point
        target_keypoint /= max_point

        input_dict = {'source_keypoint': source_keypoint,
                      'target_keypoint': target_keypoint,
                      'occlusion_label': occlusion_label,
                      'croppd_label': croppd_cond}

        return input_dict

    def __len__(self):
        return self.dataset_size
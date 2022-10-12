"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.kpe_model import KPEModel
from util.visualizer import Visualizer
from util import html
from util.util import draw_pose_from_cords
from scores.scores import score
import torch

opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = KPEModel(opt)
model.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.id,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.id, opt.phase, opt.which_epoch))


# test
for i, data_i in enumerate(dataloader):
    if i * opt.batch_size >= opt.how_many:
        break
    fake_keypoint, occlusion_pred = model(data_i, mode='inference')

    if opt.display :
        for b in range(fake_keypoint.shape[0]):
            img_path = '{0:0>3}_'.format(i)+'{0:0>6}.png'.format(b)
            print('process image... %s' % img_path)

            visuals = dict()
            visuals['source_color_map'] = model.heatmap['src_color_map'][b]
            visuals['tgt_color_map'] = model.heatmap['tgt_color_map'][b]
            visuals['fake_color_map'] = model.heatmap['fake_color_map'][b]
            visualizer.save_images(webpage, visuals, img_path)

    # Calculate score
    # * PCKh, L2, heatmap score(뭐로할지 아직 안정함)
    (l2, l2_mask), (pckh, pckh_mask), (acc, recall, precision, f1) = score(data_i['source_keypoint'] * model.max_point_tensor,
                                                                           data_i['target_keypoint'] * model.max_point_tensor,
                                                                           fake_keypoint * model.max_point_tensor,
                                                                           occlusion_pred.squeeze(),
                                                                           data_i['occlusion_label'])



webpage.save()

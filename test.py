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
from util import html, util
from util.util import draw_pose_from_cords
from scores.scores import score
import torch
import numpy as np
import pandas as pd
import key_point_name as kpn

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

l2_scores = []
l2_mask_scores = []
pckh_scores = []
pckh_mask_scores = []

accuracys = []
recalls = []
precisions = []
f1s = []

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

l2_scores.append(l2)
l2_mask_scores.append(l2_mask)
pckh_scores.append(pckh)
pckh_mask_scores.append(pckh_mask)

# occlusion scores
accuracys.append(acc)
recalls.append(recall)
precisions.append(precision)
f1s.append(f1)

l2 = np.nanmean(np.stack(l2_scores, axis=0), axis=0)
l2_mask = np.nanmean(np.stack(l2_mask_scores, axis=0), axis=0)
pckh = np.nanmean(np.stack(pckh_scores, axis=0), axis=0)
pckh_mask = np.nanmean(np.stack(pckh_mask_scores, axis=0), axis=0)
keypoint_df = pd.DataFrame([l2, l2_mask, pckh, pckh_mask],
                           columns=['total'] + kpn.key_point_name,
                           index=['l2', 'l2_masked', 'pckh', 'pckh_masked'])

accuracys = np.nanmean(np.stack(accuracys, axis=0), axis=0)
recalls = np.nanmean(np.stack(recalls, axis=0), axis=0)
precisions = np.nanmean(np.stack(precisions, axis=0), axis=0)
f1s = np.nanmean(np.stack(f1s, axis=0), axis=0)

occlusion_df = pd.DataFrame([accuracys, recalls, precisions, f1s],
                            columns=['total'] + kpn.key_point_name,
                            index=['Acc', 'Recall', 'Precision', 'F1'])
save_path = os.path.join(opt.results_dir, opt.id, 'test_scores')
util.mkdirs(save_path)
keypoint_df.to_csv(os.path.join(save_path, 'keypoint_score.csv'), index = True)
occlusion_df.to_csv(os.path.join(save_path, 'occlusion_score.csv'), index = True)
webpage.save()

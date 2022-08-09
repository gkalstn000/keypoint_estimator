import numpy as np
import torch
import torch.nn
import pandas as pd
import numpy as np
import key_point_name as kpn


def L2_score(source, target, prediction) :
    mask_point = torch.Tensor([-1, -1])
    mask_index = source == mask_point[None, None, :]
    mask_index = torch.logical_and(mask_index[:, :, 0], mask_index[:, :, 1]) * 1.0
    difference = target - prediction
    # 이 밑으로 total_score, key_point별 score, masked_total_score, masked_key_point별 score
    l2 = (difference ** 2).sum(dim = -1) ** 0.5
    l2_total = l2.mean()
    l2_key_point = l2.mean(0)

    mask_index[mask_index == 0] = float('nan')
    l2_masked = l2 * mask_index
    l2_masked_total = l2_masked.nanmean()
    l2_masked_key_point = l2_masked.nanmean(0)
    # l2_masked_key_point = torch.nan_to_num(l2_masked_key_point, 0.0)
    return [l2_total.tolist()] + l2_key_point.tolist(), \
            [l2_masked_total.tolist()] + l2_masked_key_point.tolist()

def pckh_score(source, target, prediction) :
    mask_point = torch.Tensor([-1, -1])
    mask_index = source == mask_point[None, None, :]
    mask_index = torch.logical_and(mask_index[:, :, 0], mask_index[:, :, 1]) * 1.0

    PARTS_SEL = [0, 1, 14, 15, 16, 17]
    face_points = target[:, PARTS_SEL]
    max_h, _ = face_points[:, :, 0].max(-1)
    min_h, _ = face_points[:, :, 0].min(-1)
    max_w, _ = face_points[:, :, 1].max(-1)
    min_w, _ = face_points[:, :, 1].min(-1)

    h_threshold = max_h - min_h
    w_threshold = max_w - min_w
    threshold = torch.stack([h_threshold, w_threshold], dim=1)
    difference = torch.abs(target - prediction)
    # 이 밑으로 total_score, key_point별 score, masked_total_score, masked_key_point별 score
    check_threshold = difference <= threshold.unsqueeze(1)
    pckh = torch.logical_and(check_threshold[:, :, 0], check_threshold[:, :, 1]).float()
    pckh_total = pckh.mean()
    pckh_key_point = pckh.mean(0)

    mask_index[mask_index == 0] = float('nan')
    pckh_masked_tatal = (pckh * mask_index).nanmean()
    pckh_masked_key_ponint = (pckh * mask_index).nanmean(0)

    return [pckh_total.tolist()] + pckh_key_point.tolist(), \
            [pckh_masked_tatal.tolist()] + pckh_masked_key_ponint.tolist()

def score(source, target, prediction) :
    l2, l2_mask = L2_score(source, target, prediction)
    pckh, pckh_mask = pckh_score(source, target, prediction)

    return (l2, l2_mask), (pckh, pckh_mask)


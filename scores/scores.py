import numpy as np
import torch

def L2_score(source, target, prediction) :
    mask_point = torch.Tensor([-1, -1])
    mask_index = source == mask_point[None, None, :]
    mask_index = torch.logical_and(mask_index[:, :, 0], mask_index[:, :, 1])

    norm = ((target - prediction) ** 2).sum(dim = -1) ** 0.5
    norm_masked = norm*mask_index

    total_score = norm.mean(-1)
    masked_score = norm_masked.nansum(-1) / (norm_masked != 0).nansum(-1)

    return total_score.nanmean(), masked_score.nanmean()

def pckh_score(source, target, prediction) :
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
    check_threshold = difference <= threshold.unsqueeze(1)

    return torch.logical_and(check_threshold[:, :, 0], check_threshold[:, :, 1]).float().mean()

def score(source, target, prediction) :
    pass
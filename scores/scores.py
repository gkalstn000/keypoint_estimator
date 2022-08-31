import numpy as np
import torch
import torch.nn
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import key_point_name as kpn


def L2_score(source, target, prediction, occlusion_tgt) :
    mask_point = torch.Tensor([-1, -1])
    mask_index = source == mask_point[None, None, :]
    mask_index = torch.logical_and(mask_index[:, :, 0], mask_index[:, :, 1]) * 1.0

    difference = target - prediction
    difference[occlusion_tgt == 1] = float('nan') # occlusion keypoint score에서 제외
    # 이 밑으로 total_score, key_point별 score, masked_total_score, masked_key_point별 score
    l2 = (difference ** 2).sum(dim = -1) ** 0.5
    l2_total = l2.nanmean()
    l2_key_point = l2.nanmean(0)

    mask_index[mask_index == 0] = float('nan') # mask가 아닌 keypoint score에서 제외
    l2_masked = l2 * mask_index
    l2_masked_total = l2_masked.nanmean()
    l2_masked_key_point = l2_masked.nanmean(0)
    # l2_masked_key_point = torch.nan_to_num(l2_masked_key_point, 0.0)
    return [l2_total.tolist()] + l2_key_point.tolist(), \
            [l2_masked_total.tolist()] + l2_masked_key_point.tolist()

def pckh_score(source, target, prediction, occlusion_tgt) :
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
    pckh[occlusion_tgt == 1] = float('nan')  # occlusion keypoint score에서 제외

    pckh_total = pckh.nanmean()
    pckh_key_point = pckh.nanmean(0)

    mask_index[mask_index == 0] = float('nan')
    pckh_masked_tatal = (pckh * mask_index).nanmean()
    pckh_masked_key_ponint = (pckh * mask_index).nanmean(0)

    return [pckh_total.tolist()] + pckh_key_point.tolist(), \
            [pckh_masked_tatal.tolist()] + pckh_masked_key_ponint.tolist()
def occlusion_score(occlusion_pred, occlusion_tgt) :
    occlusion_pred = (occlusion_pred.squeeze() > 0.5) * 1

    occlusion_pred_flat = occlusion_pred.view(-1, 1).squeeze()
    occlusion_tgt_flat = occlusion_tgt.view(-1, 1).squeeze()

    acc_tatal = accuracy_score(occlusion_tgt_flat, occlusion_pred_flat)
    rec_total = recall_score(occlusion_tgt_flat, occlusion_pred_flat, zero_division = 0)
    prec_total = precision_score(occlusion_tgt_flat, occlusion_pred_flat, zero_division = 0)
    f1_total = f1_score(occlusion_tgt_flat, occlusion_pred_flat, zero_division = 0)

    acc_list = []
    rec_list = []
    prec_list = []
    f1_list = []

    for true, pred in zip(occlusion_tgt.transpose(1, 0), occlusion_pred.transpose(1, 0)) :
        acc = accuracy_score(true, pred)
        rec = recall_score(true, pred, zero_division = 0)
        prec = precision_score(true, pred, zero_division = 0)
        f1 = f1_score(true, pred, zero_division = 0)

        acc_list.append(acc)
        rec_list.append(rec)
        prec_list.append(prec)
        f1_list.append(f1)

    accuracy = [acc_tatal] + acc_list
    recall = [rec_total] + rec_list
    precision = [prec_total] + prec_list
    f1 = [f1_total] + f1_list

    return accuracy, recall, precision, f1

def score(source_keypoinot, target_keypoint, prediction_keypoint, occlusion_pred, occlusion_tgt) :
    l2, l2_mask = L2_score(source_keypoinot, target_keypoint, prediction_keypoint, occlusion_tgt)
    pckh, pckh_mask = pckh_score(source_keypoinot, target_keypoint, prediction_keypoint, occlusion_tgt)
    acc, recall, precision, f1 = occlusion_score(occlusion_pred, occlusion_tgt)
    return (l2, l2_mask), (pckh, pckh_mask), (acc, recall, precision, f1)


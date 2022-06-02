import numpy as np
import torch

def score(src, tgt, pred) :
    mask_point = torch.Tensor([-1, -1])
    mask_index = src == mask_point[None, None, :]
    mask_index = torch.logical_and(mask_index[:, :, 0], mask_index[:, :, 1])

    norm = ((tgt - pred) ** 2).sum(dim = -1) ** 0.5
    norm_masked = norm*mask_index

    total_score = norm.mean(-1)
    masked_score = norm_masked.sum(-1) / (norm_masked != 0).sum(-1)

    return total_score.mean(), masked_score.mean()
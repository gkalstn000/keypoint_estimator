import numpy as np

def score(src, tgt, pred) :
    mask_index = src == np.array([-1, -1])
    mask_index = np.logical_and(mask_index[:, 0], mask_index[:, 1])

    norm = ((tgt - pred.numpy()) ** 2).sum(axis=1) ** 0.5
    norm_masked = norm[mask_index]

    total_score = norm.mean()
    masked_score = norm_masked.mean()

    return total_score / src.shape[0], masked_score / src.shape[0]
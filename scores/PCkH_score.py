MISSING_VALUE = -1

PARTS_SEL = [0, 1, 14, 15, 16, 17]
'''
  hz: head size
  alpha: norm factor
  px, py: predict coords
  tx, ty: target coords
'''

def isRight(pred, tgt, hz, alpha):
    if pred[0] == -1 or pred[1] == -1 or tgt[0] == -1 or tgt[1] == -1:
        return 0
    if abs(pred[0] - tgt[0]) < hz[0] * alpha and abs(pred[1] - tgt[1]) < hz[1] * alpha :
        return 1
    else:
        return 0


def how_many_right_seq(prediction, target, hz, alpha):
    nRight = 0
    for pred, tgt in zip(prediction, target):
        nRight = nRight + isRight(pred, tgt, hz, alpha)

    return nRight


def ValidPoints(tx):
    nValid = 0
    for item in tx:
        if item != -1:
            nValid = nValid + 1
    return nValid


def get_head_wh(tgt):
    final_w, final_h = -1, -1
    component_count = 0
    save_componets = []
    for component in PARTS_SEL:
        if tgt[component][0] == MISSING_VALUE or tgt[component][1] == MISSING_VALUE:
            continue
        else:
            component_count += 1
            save_componets.append(tgt[component].tolist())
    if component_count >= 2:
        h_cords = []
        w_cords = []
        for component in save_componets:
            h_cords.append(component[0])
            w_cords.append(component[1])
        hmin = min(h_cords)
        hmax = max(h_cords)
        wmin = min(w_cords)
        wmax = max(w_cords)

        final_w = wmax - wmin
        final_h = hmax - hmin
    return final_h, final_w

def pckh_score(source, target, prediction) :
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





def cal_limb_agreement(output, target) :
    output_directions = get_direction_vector(output)
    target_directions = get_direction_vector(target)
    agreements = (output_directions * target_directions).sum(-1) / (torch.norm(output_directions, dim = -1) * (torch.norm(target_directions, dim = -1)))
    n_loglikelihood = -torch.log(torch.abs(agreements))
    return n_loglikelihood.nansum(-1).nanmean()
def get_direction_vector(points) :
    vectors = [[1, 0], [1, 2], [1, 8], [1, 11], [1, 5], [2, 3], [3, 4], [8, 9], [9, 10], [11, 12], [12, 13], [5, 6], [6, 7], [0, 14], [14, 16], [0, 15], [15, 17]]
    return get_relationship(points, vectors)


def get_relationship(points, vector) :
    buffer = []
    for src, dest in vector :
        buffer.append(points[:, dest] - points[:, src])
    return torch.stack(buffer, dim=1)


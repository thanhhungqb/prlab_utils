import torch


def make_theta_from_st(st, is_inverse=False):
    """
    ST without rotate
    note: batch mode [-1, 4] to [-1, 2, 3]
    TODO make sure none zero for all 4 number, or replace with very small values
    :param st: [[sx, tx, sy, ty]] (omit 2 zeros) st_size
    :param is_inverse: return matrix or inverse of matrix
    :return: [[sx, 0, tx], [0, sy, ty]] or [[1/sx, 0, -tx/sx], [0, 1/sy, -ty/sy]] (reverse)
    """
    zero_vec = torch.zeros(st.size()[0], dtype=torch.float)
    if st.get_device() >= 0:
        zero_vec = zero_vec.to(device=st.get_device())

    o = [st[:, 0], zero_vec, st[:, 1], zero_vec, st[:, 2], st[:, 3]]

    if is_inverse:
        o = [1 / st[:, 0], zero_vec, -st[:, 1] / st[:, 0], zero_vec, 1 / st[:, 2], -st[:, 3] / st[:, 2]]
    o_tmp = torch.stack(o, dim=-1)

    return o_tmp.view(-1, 2, 3)

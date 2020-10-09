import numpy as np
from scipy.ndimage import rotate


def rand_crop_near_center(img_tensor, crop_size, d, **kwargs):
    """
    2D/3D rand_crop near center, combine of transform and center crop
    :param img_tensor: [z, y, x]
    :param crop_size: [sz, sy, sx] or [sy, sx], |crop_size| <= |img_tensor.shape|
    :param d: [dz, dy, dx] or [dy, dx], |d| == |crop_size|
    :return:
    """
    assert len(img_tensor.shape) in [2, 3]
    assert len(crop_size) in [2, 3] and len(crop_size) <= len(img_tensor.shape)
    assert len(d) == len(crop_size)

    crop_size = crop_size if len(crop_size) == len(img_tensor.shape) else (img_tensor.shape[0], *crop_size)
    d = d if len(d) == len(crop_size) else (0, *d)
    org_dim, dim, d = np.array(img_tensor.shape, dtype=int), np.array(crop_size, dtype=int), np.array(d, dtype=int)

    if len(img_tensor.shape) == 3 and org_dim[0] < dim[0] + 2 * d[0]:
        # padding for number of slices if less than require
        n_pad = dim[0] + 2 * d[0] - org_dim[0]
        img_tensor = np.pad(img_tensor, ((n_pad // 2, n_pad - n_pad // 2), (0, 0), (0, 0)), 'edge')
        org_dim = np.array(img_tensor.shape, dtype=int)

    assert np.all(org_dim - dim - 2 * d >= 0)

    r_d = (np.random.rand(len(d)) * d).astype(int)
    ss = (org_dim - dim) // 2 - r_d

    if len(img_tensor.shape) == 2:
        return img_tensor[ss[0]:ss[0] + dim[0], ss[1]:ss[1] + dim[1]]
    elif len(img_tensor.shape) == 3:
        return img_tensor[ss[0]:ss[0] + dim[0], ss[1]:ss[1] + dim[1], ss[2]:ss[2] + dim[2]]
    else:
        raise Exception(f"input dim should be in [2, 3], found {img_tensor.shape}")


def random_rotate_xy(img_numpy, angle=(-30, 30), **kwargs):
    """
    random rotate slices by in plate xy (omit z) in form of input [z, y, x].
    NOTE: the output shape should be different with input shape, a crop may be need after this step
    :param img_numpy: numpy [z, y, x] or [y, x]
    :param angle: -360<=vals<=360 (min, max)
    :param kwargs:
    :return: random rotate img
    """
    min_angle, max_angle = angle
    assert img_numpy.ndim in [2, 3], "provide a 2d/3d numpy array"
    assert min_angle < max_angle, "min should be less than max val"
    assert min_angle > -360 or max_angle < 360

    angle = np.random.randint(low=min_angle, high=max_angle + 1)
    axes = (1, 2) if img_numpy.ndim == 3 else (0, 1)  # img in form: [n_slices, y, x]
    return rotate(img_numpy, angle, axes=axes)

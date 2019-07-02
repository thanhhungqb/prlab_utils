from pathlib import Path

import PIL
import numpy as np
import torch
from fastai.core import ItemBase
from fastai.vision import ImageList, open_image, Image, crop_pad, pil2tensor
from torch import Tensor

"""
This file implement classes and function to apply to video that can be load and
word with fastai
"""


def next_k_frames_file(fpath, k=30, prefix="frame"):
    """
    Next k frame file names with form: {prefix}{number}.{suffix}
    :param fpath:
    :param k:
    :param prefix:
    :return: [k path]
    """
    fpath = fpath if isinstance(fpath, Path) else Path(fpath)
    fparent = fpath.parent
    n = fpath.stem
    it = int(n[len(prefix):])
    atl = [fparent / ("{}{}{}".format(prefix, it + i, fpath.suffix)) for i in range(k)]
    return atl


class VideoFramesItem(ItemBase):
    """
    Contains list of frames
    """

    def __init__(self, *imgs):
        self.imgs = list(imgs)
        self.obj, self.data = imgs, torch.stack([img.data for img in self.imgs])

    def apply_tfms(self, tfms, **kwargs):
        for i in range(len(self.imgs)):
            self.imgs[i] = self.imgs[i].apply_tfms(tfms, **kwargs)

        self.data = torch.stack([img.data for img in self.imgs])
        return self

    def __repr__(self) -> str: return f'{self.__class__.__name__} {self.data.size()}'


class VideoFramesList(ImageList):
    """
    Work with Video frames in each sub-folder, each frame have name {prefix}{number}.{suffix}
    Load multi frames {n_frame} to work as the same time.
    """

    def __init__(self, items, n_frame=5, prefix='frame', **kwargs):
        super().__init__(items, **kwargs)
        self.n_frame = n_frame
        self.prefix = prefix

        [self.copy_new.append(o) for o in ['n_frame', 'prefix']]

    def set_n_frame(self, n_frame=5):
        """
        Set n_frame
        :param n_frame:
        :return:
        """
        self.n_frame = n_frame
        return self

    def get(self, i):
        # img0 = super().get(i)
        item = self.items[i]

        names = next_k_frames_file(item, k=self.n_frame, prefix=self.prefix)

        # if outside of files list (repeat latest frame)
        is_file = [o.is_file() for o in names]
        if not is_file[0]:
            names[0] = item
        for i in range(1, len(names)):
            if not is_file[i]:
                names[i] = names[i - 1]  # copy from latest frame

        imgs = [open_image(fn) for fn in names]

        res = VideoFramesItem(*imgs)

        return res

    @classmethod
    def from_folders(cls, path, **kwargs):
        res = super().from_folder(path, **kwargs)
        res.path = path
        return res

    def reconstruct(self, t: Tensor):
        imgs = [Image(t[i]) for i in range(self.n_frame)]
        return VideoFramesItem(*imgs)


class CropImageList(ImageList):
    """
    Load and crop image on the fly.
    Use Dataframe index: name, bb: [left top right bottom]
    """

    def __init__(self, items, df=None, scale_type='fixed', max_scale=1.2, min_scale=1., **kwargs):
        super().__init__(items, **kwargs)
        self.df = df

        self.scale_type = scale_type
        self.max_scale = max_scale
        self.min_scale = min_scale

        [self.copy_new.append(o) for o in ['df', 'scale_type', 'max_scale', 'min_scale']]

    def set_values(self, df=None, scale_type=None, max_scale=None, min_scale=None):
        """
        Set values and return self to easy to chain
        :param df:
        :param scale_type:
        :param max_scale:
        :param min_scale:
        :return:
        """
        if df is not None:
            self.df = df

        if scale_type is not None:
            self.scale_type = scale_type

        if max_scale is not None:
            self.max_scale = max_scale

        if min_scale is not None:
            self.min_scale = min_scale

        return self

    def get(self, i):
        img = super().get(i)
        item = self.items[i]

        # Crop by info from self.df
        item_path = item if isinstance(item, Path) else Path(item)
        img_width, img_heigh = img.size

        # select area to crop from df['bb']
        center_x, center_y, dx, dy = self.df.loc[item_path.name, 'bb']

        s_zoom = np.random.uniform(self.min_scale, self.max_scale, 4) \
            if self.scale_type == 'random' else [self.max_scale] * 4

        left, right = center_x - dx / 2 * s_zoom[0], center_x + dx / 2 * s_zoom[1]
        top, bottom = center_y - dy / 2 * s_zoom[2], center_y + dy / 2 * s_zoom[3]

        # refine if out of size
        # left, top, right, bottom = max(1, left), max(1, top), min(img_width - 1, right), min(img_heigh - 1, bottom)
        size = int(bottom - top), int(right - left)
        center = (right + left) / 2 / img_width, (bottom + top) / 2 / img_heigh

        # cropped = img.crop((left, top, right, bottom))
        cropped = crop_pad(img, size=size, row_pct=center[0], col_pct=center[1])

        self.sizes[i] = cropped.size
        return cropped


class YCbCrImageList(ImageList):
    """
    three channel of YCbCr Image (instead RGB as normal)
    """

    def __init__(self, items, **kwargs):
        super(YCbCrImageList, self).__init__(items, **kwargs)

    def get(self, i):
        fn = self.items[i]

        img = PIL.Image.open(fn).convert('YCbCr')
        channel3 = pil2tensor(img, np.float32)
        channel3.div_(255)

        return Image(channel3)

from pathlib import Path

from fastai.core import ItemBase
from fastai.vision import ImageList, open_image, Image
from torch import Tensor

"""
This file implement classes and function to apply to video that can be load and
word with fastai
"""


def next_k_frames_file(fpath, k=5, prefix="frame"):
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
        self.obj, self.data = imgs, [img.data for img in self.imgs]

    def apply_tfms(self, tfms, **kwargs):
        for i in range(len(self.imgs)):
            self.imgs[i] = self.imgs[i].apply_tfms(tfms, **kwargs)

        self.data = [img.data for img in self.imgs]
        return self

    def __repr__(self) -> str: return f'{self.__class__.__name__}'


class VideoFramesList(ImageList):
    """
    Work with Video frames in each sub-folder, each frame have name {prefix}{number}.{suffix}
    Load multi frames {n_frame} to work as the same time.
    """

    def __init__(self, items, n_frame=5, prefix='frame', **kwargs):
        super().__init__(items, **kwargs)
        self.n_frame = n_frame
        self.prefix = prefix

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

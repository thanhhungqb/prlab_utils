from fastai.vision import *
from torch.utils.data import SequentialSampler

"""
This file implement classes and function to apply to video that can be load and
word with fastai
"""


def convert_to_fastai_crop_format(left, top, w, h, img_width, img_height):
    """
    :param left: left to crop
    :param top:  top of crop
    :param w:  width of crop
    :param h:  height of crop
    :param img_width: original image width
    :param img_height: original image height
    :return: (size, row_pct, col_pct) used to pass to fastai.vision.crop
    """
    right, bottom = left + w, top + h
    left, top, right, bottom = max(1., left), max(1., top), min(img_width - 1, right), min(img_height - 1, bottom)
    center = round(left + right) // 2, round(top + bottom) // 2
    size = round(right - left), round(bottom - top)
    # old_bb = np.array(self.df.loc[self.items[i].name, 'bb'])
    # self._new_center[i] = old_bb[:2] - np.array([left, top])

    # self._new_center[i] = self._new_center[i][0], self._new_center[i][1], old_bb[2], old_bb[3]

    row_pct = (center[1] - size[1] / 2) / (img_height - size[1]) if img_height > size[1] else 0.5
    col_pct = (center[0] - size[0] / 2) / (img_width - size[0]) if img_width > size[0] else 0.5
    size = size[1], size[0]  # change to rows, cols

    return size, row_pct, col_pct


def crop_img(img_path, size, row_pct, col_pct):
    # convert bbox to format of crop
    img = open_image(img_path)

    cropped = crop(img, size=size, row_pct=row_pct, col_pct=col_pct)

    return cropped


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
    TODO wrong implement of crop, see `SizeGroupedCropImageList` to fix
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

    def set_values(self, df=None, scale_type=None, max_scale=None, min_scale=None, bunch=None):
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

        if bunch is not None:
            self._bunch = bunch

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


# Related collate function
# TODO now hard code to convert Category to int, need find the way to smarter
# TODO need limit maximum size to 512? and related to bs
def get_size_rec(b: ImageList, f_size=np.max):
    """
    Get biggest size to resize for all in batch
    :param b:
    :param f_size:
    :return:
    """
    if is_listy(b): return f_size([get_size_rec(o) for o in b])
    return f_size(b.size) if isinstance(b, Image) else 0


def to_data_resize(b: ImageList, n_size=None):
    """
    Must resize to same size before convert to Tensor
    Recursively map lists of items in `b ` to their wrapped data.
    """
    # resize to same size
    if n_size is None:
        n_size = get_size_rec(b)
    n_size = int(n_size)
    n_size = 224 if n_size > 224 else n_size

    if is_listy(b):
        out = [to_data_resize(o, n_size) for o in b]
        tmp = []
        for i in range(len(out) // 2):
            if out[i * 2] is not None and out[i * 2 + 1] is not None:
                tmp.append(out[i * 2])
                tmp.append(out[i * 2 + 1])
        out = tmp
        return out

    try:
        out = b.resize(size=n_size).data if isinstance(b, Image) else int(b)  # hard code at int, TODO check way to pass
    except Exception as e:
        print(e)
        print(n_size, b)
        return None
    return out


def resize_collate(batch: ItemsList) -> Tensor:
    """Convert `batch` items to tensor data. see `data_collate`"""
    return torch.utils.data.dataloader.default_collate(to_data_resize(batch))


# -----------------------------------------------------------------------------


class GroupRandomSampler(SequentialSampler):
    """
    implement random sampler in each groups, RandomSampler
    need call new_crop_info and ... to update sampler and groups
    work with SizeGroupedCropImageList
    """

    def __iter__(self):
        if isinstance(self.data_source, SizeGroupedImageDataBunch):
            self.data_source.size_group()  # make new random if have

        return iter(self.data_source._sampler)


class SizeGroupedImageDataBunch(ImageDataBunch):
    """
    TODO note set size=None and apply resize after (in batch mode via collate?)
    """

    @classmethod
    def create(cls, train_ds: Dataset, valid_ds: Dataset, test_ds: Optional[Dataset] = None, path: PathOrStr = '.',
               bs: int = 64,
               val_bs: int = None, num_workers: int = defaults.cpus, dl_tfms: Optional[Collection[Callable]] = None,
               device: torch.device = None, collate_fn: Callable = resize_collate, no_check: bool = False,
               **dl_kwargs) -> 'DataBunch':
        """Create a `DataBunch` from `train_ds`, `valid_ds` and maybe `test_ds` with a batch size of `bs`.
        Passes `**dl_kwargs` to `DataLoader()`"""
        datasets = cls._init_ds(train_ds, valid_ds, test_ds)
        val_bs = ifnone(val_bs, bs)

        collate_fn = resize_collate  # TODO fix hardcoded use param instead

        train_sampler = GroupRandomSampler(datasets[0].x)
        train_dl = DataLoader(datasets[0], batch_size=bs, sampler=train_sampler, drop_last=True,
                              num_workers=num_workers, **dl_kwargs)
        dataloaders = [train_dl]

        for ds in datasets[1:]:
            sampler = GroupRandomSampler(ds.x)
            dataloaders.append(DataLoader(ds, batch_size=val_bs, sampler=sampler, num_workers=num_workers, **dl_kwargs))
        return cls(*dataloaders, path=path, device=device, dl_tfms=dl_tfms, collate_fn=collate_fn, no_check=no_check)


class SizeGroupedCropImageList(CropImageList):
    """
    group by size (max w, h), number of group default is 10 (can be change)
    bunch can make difference input size (image) on difference batch
    """
    _bunch = SizeGroupedImageDataBunch
    _sampler = None
    _n_group = 10
    _groups = {}
    _crop_info = None
    _new_center = {}
    _resize_method = [ResizeMethod.CROP, ResizeMethod.PAD, ResizeMethod.SQUISH][2]  # choose 1 or 2

    def get(self, pos):

        if self._sampler is None:
            self.size_group()

        i = pos  # self._sampler[pos]  # get by sampler not the order in items

        fn = self.items[i]
        img = self.open(fn)

        img_heigh, img_width = img.size

        center, size = self._crop_info[i]

        # refine size and center to not outside image
        left, right = center[0] - size[0] / 2, center[0] + size[0] / 2
        top, bottom = center[1] - size[1] / 2, center[1] + size[1] / 2
        left, top, right, bottom = max(1., left), max(1., top), min(img_width - 1, right), min(img_heigh - 1, bottom)
        center = round(left + right) // 2, round(top + bottom) // 2
        size = round(right - left), round(bottom - top)
        old_bb = np.array(self.df.loc[self.items[i].name, 'bb'])
        self._new_center[i] = old_bb[:2] - np.array([left, top])

        self._new_center[i] = self._new_center[i][0], self._new_center[i][1], old_bb[2], old_bb[3]

        row_pct = (center[1] - size[1] / 2) / (img_heigh - size[1]) if img_heigh > size[1] else 0.5
        col_pct = (center[0] - size[0] / 2) / (img_width - size[0]) if img_width > size[0] else 0.5
        size = size[1], size[0]  # change to rows, cols
        cropped = crop(img, size=size, row_pct=row_pct, col_pct=col_pct)

        # padding to same size max(w, h)
        nsize = size[0] if size[0] > size[1] else size[1]
        cropped = cropped.apply_tfms([crop_pad()], size=nsize, resize_method=self._resize_method, padding_mode='zeros')

        self.sizes[i] = cropped.size
        return cropped

    def size_group(self, n_group=None):
        """
        call for the first get command or outside to make new order (in case random)
        :param n_group:
        :return:
        """
        if n_group is not None:
            self._n_group = n_group

        self.new_crop_info()

        s_size = np.array([np.max(list(o)) for c, o in self._crop_info.items()])

        groups = {i: [] for i in range(self._n_group)}
        vv = [np.percentile(s_size, 100 * i / self._n_group) for i in range(1, self._n_group + 1)]
        for pos in range(len(self.items)):
            _, size = self._crop_info[pos]

            p_group = self._n_group - 1
            for i in range(self._n_group):
                n_size = np.max(size)
                if n_size <= vv[i]:
                    p_group = i
                    break
            groups[p_group].append(pos)

        self._groups = groups
        self._sampler = np.concatenate([np.random.permutation(val) for k, val in groups.items()]).astype(np.int)
        return self

    def crop_calc(self, i):
        """
        Calculate and store center, (w,h) to crop for items
        :param i: position in items
        :return:
        """
        item = self.items[i]

        # Crop by info from self.df
        item_path = item if isinstance(item, Path) else Path(item)

        # select area to crop from df['bb']
        center_x, center_y, dx, dy = self.df.loc[item_path.name, 'bb']

        s_zoom = np.random.uniform(self.min_scale, self.max_scale, 4) \
            if self.scale_type == 'random' else [self.max_scale] * 4

        left, right = center_x - dx / 2 * s_zoom[0], center_x + dx / 2 * s_zoom[1]
        top, bottom = center_y - dy / 2 * s_zoom[2], center_y + dy / 2 * s_zoom[3]

        # refine if out of size
        # left, top, right, bottom = max(1, left), max(1, top), min(img_width - 1, right), min(img_heigh - 1, bottom)
        size = int(right - left), int(bottom - top)  # w, h
        center = int(right + left) // 2, int(bottom + top) // 2
        return center, size

    def new_crop_info(self):
        """
        update a new resize for items
        :return:
        """
        self._crop_info = {}
        for i in range(len(self.items)):
            self._crop_info[i] = self.crop_calc(i)


class ImageTuple(ItemBase):
    def __init__(self, img1, img2):
        self.img1, self.img2 = img1, img2
        self.obj, self.data = (img1, img2), [img1.data, img2.data]

    def apply_tfms(self, tfms, **kwargs):
        """
        use tfms2 in kwargs for img2 if found
        :param tfms:
        :param kwargs:
        :return:
        """
        self.img1 = self.img1.apply_tfms(tfms, **kwargs)
        self.img2 = self.img2.apply_tfms(kwargs.get('tfms2', tfms), **kwargs)
        self.data = [self.img1.data, self.img2.data]
        return self

    def to_one(self): return Image(torch.cat(self.data, 2))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__} {list(self.data[0].size())} {list(self.data[1].size())}'


class SelfRepresentImageList(ImageList):
    """
    label_func like that: label_func = lambda o: 0 if target_func1(o[0]) == target_func1(o[1]) else 1
    """

    def __init__(self, base_list, pos=None, **kwargs):
        kwargs['items'] = base_list.items
        super().__init__(**kwargs)
        # super().__init__(**kwargs)

        self.base_list = base_list

        self.pos = np.random.permutation(len(base_list.items))

        [self.copy_new.append(o)
         for o in ['base_list', 'pos']]

    def get(self, i):
        img1 = self.base_list.get(i)
        img2 = self.base_list.get(self.pos[i])
        return ImageTuple(img1, img2)

    def label_from_func(self, func: Callable, label_cls: Callable = None, **kwargs) -> 'LabelList':
        "Apply `func` to every input to get its label."
        self.pos = np.random.permutation(len(self.items))
        itemsB = [self.items[o] for o in self.pos]

        return self._label_from_list([func(o) for o in zip(self.items, itemsB)], label_cls=label_cls, **kwargs)

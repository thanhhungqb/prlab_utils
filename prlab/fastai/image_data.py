"""
Implement dataload, process for Image data to use in fastai
"""
from fastai.vision import *
from torch.utils.data import RandomSampler

# ************************************************
from prlab.gutils import load_func_by_name


class ClassBalancedRandomSampler(RandomSampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, TODO note, items maybe occur more than 1 times.
    Mostly use with classification problem with high unbalanced labels dataset
    Arguments:
        data_source (Dataset): dataset to sample from
        max_samples_each_class (int): number of samples for each class
        replacement (bool): samples are drawn with replacement if ``True``, default=``False``
    """

    def __init__(self, data_source, max_samples_each_class, data_helper, replacement=False, **config):
        # must provide the way to know the label?
        self.data_source = data_source
        self.replacement = replacement
        self.max_samples_each_class = max_samples_each_class

        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

        self.data_helper = data_helper
        self.selected_pos = None

        self.resample()

    def resample(self):
        # select and store all info here (more memory need to store this list, but less complex to implement)
        # Do different for replacement or not
        # TODO now not support replacement
        # n = len(self.data_source)
        labels = [self.data_helper.y_func(item) for item in self.data_source.items]
        pos = [o for o in range(len(labels))]
        random.shuffle(pos)
        self.selected_pos = []
        count_l = {}
        for p in pos:
            label_p = labels[p]
            if count_l.get(label_p, 0) < self.max_samples_each_class:
                count_l[label_p] = count_l.get(label_p, 0) + 1
                self.selected_pos.append(p)

    @property
    def num_samples(self):
        return len(self.selected_pos)

    def __iter__(self):
        self.resample()
        return iter(self.selected_pos)

    def __len__(self):
        return len(self.selected_pos)


class SamplerSuper:
    """
    A supper class to keep the way to make a sampler
    Usage: see `prlab.fastai.image_data.SamplerImageDataBunch` and `prlab.fastai.image_data.SamplerImageList`
        SamplerImageList.from_...()
            .***
            .databunch(bs=config['bs'], ..., sampler_super=SamplerSuper(**config))

        config must include sample_cls and maybe other params for sample_cls(...) as data_helper
    """

    def __init__(self, **config):
        self.config = config
        self.sample_cls = config.get('sample_cls', None)
        if self.sample_cls is not None:
            self.sample_cls = load_func_by_name(self.sample_cls)[0]
        else:
            raise Exception('Must provide sample_cls as str/obj'
                            'ex. prlab.fastai.image_data.ClassBalancedRandomSampler')

    def __call__(self, data_source, **kwargs):
        return self.make_sampler(data_source, **kwargs)

    def make_sampler(self, data_source, **kwargs):
        params = self.config.copy()
        params.update(**kwargs)

        return self.sample_cls(data_source=data_source, **params)


# ************************************************

class SamplerImageDataBunch(ImageDataBunch):
    """
    Image Data Bunch with configurable Sampler via a sampler_super.
    It can keep more than a simple data_source to make sampler more complex cases
    If skip sampler_super then it equals to ImageDataBunch, then, can use it to safe replace ImageDataBunch
    How to use:
        make a new class
            class ExtendImageList(ImageList):
                _bunch = SamplerImageDataBunch

        sampler_super = SamplerSuper(...) or any type of Sampler maker
        ExtendImageList.**
            .transform(...)
            .databunch(bs=config['bs'], ..., sampler_super=sampler_super, ...)

        see more `fastai.data_block.LabelLists.databunch` and `prlab.fastai.video_data.SizeGroupedCropImageList`
    """

    @classmethod
    def create(cls, train_ds: Dataset, valid_ds: Dataset, test_ds: Optional[Dataset] = None, path: PathOrStr = '.',
               bs: int = 64,
               val_bs: int = None, num_workers: int = defaults.cpus, dl_tfms: Optional[Collection[Callable]] = None,
               device: torch.device = None, collate_fn: Callable = data_collate, no_check: bool = False,
               sampler_super=None,
               **dl_kwargs) -> 'DataBunch':
        """Create a `DataBunch` from `train_ds`, `valid_ds` and maybe `test_ds` with a batch size of `bs`.
        override super with addition param: sampler_super, if none to equal to super class
        Passes `**dl_kwargs` to `DataLoader()`"""
        # sampler_super is a meta object can use to call to make a sampler (provided data_source)
        datasets = cls._init_ds(train_ds, valid_ds, test_ds)
        val_bs = ifnone(val_bs, bs)

        dls = []
        for d, b, s in zip(datasets, (bs, val_bs, val_bs, val_bs), (True, False, False, False)):
            if d is not None:
                if sampler_super is not None:
                    sampler = sampler_super(d.x)
                    dls.append(DataLoader(d, b, sampler=sampler, drop_last=s, num_workers=num_workers, **dl_kwargs))
                else:
                    # random for train and sequential for valid/test as default in original of ImageDataBunch
                    dls.append(DataLoader(d, b, shuffle=s, drop_last=s, num_workers=num_workers, **dl_kwargs))

        return cls(*dls, path=path, device=device, dl_tfms=dl_tfms, collate_fn=collate_fn, no_check=no_check)


# ****************************************************
class SamplerImageList(ImageList):
    """
    Extend ImageList to custom databunch, and to sampler
    """
    _bunch = SamplerImageDataBunch

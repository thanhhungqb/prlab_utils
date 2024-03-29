from builtins import super, Exception
from functools import partial
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from fastai.callback.core import Callback
from fastai.callback.progress import CSVLogger
from fastai.callback.tracker import SaveModelCallback
from fastai.interpret import ClassificationInterpretation
from fastai.learner import Learner
from fastai.metrics import accuracy, top_k_accuracy
from fastai.vision import models
from fastai.vision.core import imagenet_stats
from torch.autograd import Variable
from torch.nn.functional import log_softmax

from outside.scikit.plot_confusion_matrix import plot_confusion_matrix
from prlab.common.dl import general_dl_make_up
from prlab.common.utils import lazy_object_fn_call


def freeze_layer(x, flag=True):
    """
    :param x: layer
    :param flag: true => freezee, false => trainable
    :return:
    """
    for ele in x.parameters():
        ele.requires_grad = not flag


class SaveModelCallbackTTA(SaveModelCallback):
    """A `SaveModelCallbackTTA` that saves the model when monitored quantity is best."""

    def __init__(self, learn: Learner, monitor: str = 'accuracy', mode: str = 'auto', every: str = 'improvement',
                 name: str = 'bestmodel_tta', step: int = 3, scale=1.1, monitor_func=accuracy, is_mul=True):
        super().__init__(learn, monitor=monitor, mode=mode, every=every, name=name)
        self.step = step
        self.scale = scale
        self.monitor_func = monitor_func
        self.is_mul = is_mul
        self.f = np.min if 'loss' in monitor else np.max

    def on_epoch_end(self, epoch: int, **kwargs: Any) -> None:
        """Compare the value monitored to its best score and maybe save the model."""

        if epoch % self.step == 0:
            if self.every == "epoch":
                self.learn.save(f'{self.name}_{epoch}')
            else:  # every="improvement"
                val, tta_val, *_ = test_image_summary(self.learn, scale=self.scale, monitor_func=self.monitor_func,
                                                      k_tta=1,
                                                      is_plt=False)
                current = self.f(tta_val)

                if current is not None and self.operator(current, self.best):
                    print(f'Better model found at epoch {epoch} with {self.monitor} value: {current}.')
                    self.best = current
                    self.learn.save(f'{self.name}_{epoch}_{current}') if self.is_mul else None
                    self.learn.save(f'{self.name}')


class CheckpointLossVarianceCallback(SaveModelCallback):
    """TODO not finish yet. run get_preds many times, save monitor values and calculate variance
    compare with latest variance and choose the best (lower or higher)
    Callback that saves the model when monitored quantity variance (in some TTA) is best (lower or higher dependent
    on monitor."""

    def __init__(self, learn: Learner, monitor: str = 'accuracy', mode: str = 'max', every: str = 'improvement',
                 name: str = 'bestmodel_variance', step: int = 1, scale=1., monitor_func=accuracy, is_mul=False):
        super().__init__(learn, monitor=monitor, mode=mode, every=every, name=name)
        self.step = step
        self.scale = scale
        self.monitor_func = monitor_func
        self.is_mul = is_mul
        self.n_times = 8

    def on_epoch_end(self, epoch: int, **kwargs: Any) -> None:
        """Compare the value monitored to its best score and maybe save the model."""

        if epoch % self.step == 0:
            if self.every == "epoch":
                self.learn.save(f'{self.name}_{epoch}')
            else:  # every="improvement"
                tta_val = []
                for i in range(self.n_times):
                    preds, ys, losses = self.learn.get_preds(with_loss=True)
                    tta_val.append(accuracy(preds, ys))
                    # TODO now only support accuracy, need support loss later

                m, val = np.mean(tta_val), np.std(tta_val)
                current = np.mean(tta_val) - np.std(tta_val)

                if current is not None and self.operator(current, self.best):
                    print(f'Better model found at epoch {epoch} with {self.monitor} value: {current}.')
                    self.best = current
                    (Path(self.learn.model_dir) / 'var.log').open('a').write(f'{epoch} {self.best} ({m} + {val}) \n')

                    self.learn.save(f'{self.name}_{epoch}_{current}') if self.is_mul else \
                        self.learn.save(f'{self.name}')


class CheckpointBestUACallback(SaveModelCallback):
    """TODO not finish yet. run get_preds many times, save monitor values and calculate variance
    compare with latest variance and choose the best (lower or higher)
    Callback that saves the model when monitored quantity variance (in some TTA) is best (lower or higher dependent
    on monitor."""

    def __init__(self, learn: Learner, name: str = 'bestmodel_variance', step: int = 1, scale=1., is_mul=False):
        super().__init__(learn, monitor='accuracy', mode='max', every='improvement', name=name)
        self.monitor = 'accuracy'
        self.step = step
        self.scale = scale
        self.monitor_func = accuracy
        self.is_mul = is_mul
        self.n_times = 4

    def on_epoch_end(self, epoch: int, **kwargs: Any) -> None:
        """Compare the value monitored to its best score and maybe save the model."""

        if epoch % self.step == 0:
            if self.every == "epoch":
                self.learn.save(f'{self.name}_{epoch}')
            else:  # every="improvement"
                tta_val = []
                for i in range(self.n_times):
                    preds, ys, losses = self.learn.get_preds(with_loss=True)
                    # acc = accuracy(preds, ys)
                    preds, ys = preds.numpy(), ys.numpy()
                    preds = preds.argmax(axis=1)

                    # for label in range(4):
                    #     print('check', acc, (ys == label), (preds == ys),
                    #           (preds == ys)[ys == label], ys)
                    a = [(preds == ys)[ys == label].sum() / (ys == label).sum() for label in range(4)]
                    tta_val.append(np.mean(a))

                m, val = np.mean(tta_val), np.std(tta_val)
                current = m

                if current is not None and self.operator(current, self.best):
                    print(f'Better model found at epoch {epoch} with {self.monitor} value: {current}.')
                    self.best = current
                    (Path(self.learn.model_dir) / 'var.log').open('a').write(f'{epoch} {self.best} ({m} + {val}) \n')

                    self.learn.save(f'{self.name}_{epoch}_{current}') if self.is_mul else \
                        self.learn.save(f'{self.name}')


class ECSVLogger(CSVLogger):
    """Extend CSVLogger, flush each epoch to easy track"""

    def __init__(self, learn: Learner, filename: str = 'history', append: bool = False):
        super(ECSVLogger, self).__init__(learn, filename, append)

    def on_epoch_end(self, epoch: int, smooth_loss: torch.Tensor, last_metrics, **kwargs: Any) -> bool:
        """ fastai v1, last_metrics: MetricsList"""
        super().on_epoch_end(epoch, smooth_loss, last_metrics, **kwargs)
        self.file.flush()  # to make sure write to disk


class DataArgCallBack(Callback):
    """partial(DataArgCallBack, src=src, label_func=label_func, config=config)
    to persitent of valid, src is after split valid
    label_from_func do the random new item order (second list?)
    Use with `LRModel`
    """

    def __init__(self, learn: Learner, src, label_func, config, transform_size=None, normalize=imagenet_stats,
                 **kwargs):
        super(DataArgCallBack, self).__init__()
        self.learn = learn
        self.src = src
        self.label_func = label_func
        self.config = config
        self.transform_size = transform_size
        self.normalize = normalize

    def on_epoch_end(self, epoch: int, smooth_loss: torch.Tensor, last_metrics, **kwargs: Any) -> bool:
        # do with data
        """ fastai v1, last_metrics: MetricsList"""
        n_data = (self.src
                  .label_from_func(self.label_func)
                  .transform([[], []], size=self.transform_size)
                  .databunch(bs=self.config['bs'])
                  )

        if self.normalize:
            n_data = n_data.normalize(self.normalize)

        self.learn.data = n_data


def get_callbacks(best_name='best', monitor='valid_loss', csv_filename='log', csv_append=True):
    out = [
        partial(SaveModelCallback, monitor=monitor, name=best_name),
        callbacks.TrackerCallback,
        partial(ECSVLogger, filename=csv_filename, append=csv_append)
    ]
    return out


def difficult_weight_loss(pred, target, *args, **kwargs):
    """
    loss = - 1 * log(pi) * (pj/pi), where pj is max(p_)
    :param pred:
    :param target:
    :param args:
    :param kwargs:
    :return:
    """
    softmax = log_softmax(pred, 1)
    target_one_hot = torch.zeros(len(target.cpu()), 7).scatter_(1, target.cpu().unsqueeze(1), 1.)

    xa, _ = torch.max(softmax, dim=1)

    a = -target_one_hot.cuda() * softmax
    a = torch.sum(a, dim=1)

    xaa = 1 + xa + a
    a = a * xaa
    return torch.mean(a)


dificult_weight_loss = difficult_weight_loss


def prob_acc(pred, target, **kwargs):
    """
    accuracy when y input as raw probability instead a int number
    use when prob is not 1/0
    :param pred: raw_score
    :param target: [0.7 0.2 0.1]
    :return: accuracy
    """
    return accuracy(pred, torch.argmax(target, dim=1))


def joint_acc(preds, target, **kwargs):
    return accuracy(preds[0], target)


def make_one_hot(labels, C=2):
    """
    credit: https://gist.github.com/jacobkimmel/4ccdc682a45662e514997f724297f39f
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    """
    one_hot = torch.cuda.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.data, 1)

    target = Variable(target)

    return target


def dice_loss(input, target):
    """
    credit: https://github.com/rogertrullo/pytorch/blob/rogertrullo-dice_loss/torch/nn/functional.py#L708
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input
    """
    target = make_one_hot(target, C=2)
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 4, "Input must be a 4D Tensor."
    uniques = np.unique(target.cpu().numpy())
    assert set(list(uniques)) <= set([0, 1]), "target must only contain zeros and ones"

    probs = torch.softmax(input, dim=-1)
    num = probs * target  # b,c,h,w--p*g
    num = torch.sum(num, dim=3)  # b,c,h
    num = torch.sum(num, dim=2)

    den1 = probs * probs  # --p^2
    den1 = torch.sum(den1, dim=3)  # b,c,h
    den1 = torch.sum(den1, dim=2)

    den2 = target * target  # --g^2
    den2 = torch.sum(den2, dim=3)  # b,c,h
    den2 = torch.sum(den2, dim=2)  # b,c

    dice = 2 * (num / (den1 + den2))
    dice_eso = dice[:, 1:]  # we ignore bg dice val, and take the fg

    # dice_total = -1 * torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz
    dice_total = 1 - torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz

    return dice_total


class VAEJoinTaskMetric:
    """
    metric for VAE join task.
    the first output of VAE is for unsupervised, second output should be get to calc
    """
    __name__ = 'VAEJoinTaskMetric'

    def __init__(self, base_metric=None, **config):
        self.base_metrics = base_metric if base_metric is not None else accuracy
        self.base_metrics = lazy_object_fn_call(self.base_metrics, **config)

    def __call__(self, pred, target, *args, **kwargs):
        _, _, _, *o = pred if isinstance(pred, tuple) else (None, None, None, [pred])
        others = o[0]
        c_out = others[0]
        return self.base_metrics(c_out, target)

    def __repr__(self):
        return "VAEJoinTaskMetric with base_metrics ({})".format(str(self.base_metrics))


def test_image_summary(learn, data_test=None, scale=1.1, is_normalize=True, monitor_func=accuracy, k_tta=3,
                       is_plt=True):
    """
    Summary result with learn. Use on ipynb.
    Test data must on "valid" of data_set, else use Valid
    :param learn:
    :param data_test:
    :param scale: image scale, default is 1.1
    :param is_normalize: if want normalize confusion matrix
    :param monitor_func: monitor function, default is accuracy
    :param k_tta: number of tta
    :param is_plt: plot or not
    :return:
    """
    data = learn.data  # backup
    if data_test is not None:
        learn.data = data_test
    interp = ClassificationInterpretation.from_learner(learn)

    try:
        if is_plt:
            interp.plot_confusion_matrix(figsize=(10, 10), dpi=60, normalize=is_normalize)
            plt.show()
    except Exception as e:
        print(e)

    preds, y = learn.get_preds(with_loss=False)
    monitor_val = monitor_func(preds, y)

    tta_monitor_val = []
    preds_y = []
    for _ in range(k_tta):
        ys, y = learn.TTA(ds_type=DatasetType.Valid, scale=scale)
        preds_y.append((ys, y))
        tta_monitor_val.append(monitor_func(ys, y))

    # restore data (maybe change to data_test) then make sure as before call this function
    learn.data = data
    return monitor_val, tta_monitor_val, preds_y


def run_learner_report(learn, data=None, data_test=None, class_names=None, ret_only_fig=True):
    """

    :param learn: fastai learn
    :param data: for valid step
    :param data_test: for test step, key still valid
    :param class_names: numpy array of string
    example np.array(['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'])
    :param ret_only_fig: True => only o, else return all acc
    :return:
    """
    learn.data = data
    preds, y, losses = learn.get_preds(with_loss=True)
    ys, y = learn.TTA(ds_type=DatasetType.Valid, scale=1.1)
    acc = accuracy(ys, y)
    valid_acc1 = accuracy(preds, y)
    valid_acc = acc

    learn.data = data_test
    preds, y, losses = learn.get_preds(with_loss=True)
    ys, y = learn.TTA(ds_type=DatasetType.Valid, scale=1.1)
    acc = accuracy(ys, y)

    print('Valid acc', valid_acc1)
    print('Valid acc TTA', valid_acc)

    test_acc1 = accuracy(preds, y)
    test_acc = acc
    print('Test acc', test_acc1)
    print('Test acc TTA', test_acc)

    ys1 = torch.argmax(ys, dim=1)
    o = plot_confusion_matrix(y, ys1, classes=class_names, normalize=True)
    return o if ret_only_fig else (valid_acc1, valid_acc, test_acc1, test_acc, o)


def general_configure(**config):
    config = general_dl_make_up(**config)
    config.update({
        'tfms': get_transforms_wrap(xtra_tfms=[], **config),
        'callback_fn': lambda: get_callbacks(best_name=config['best_name'], csv_filename=config['csv_log']),

    })
    return config


def get_transforms_wrap(do_flip: bool = True, flip_vert: bool = False, max_rotate: float = 10., max_zoom: float = 1.1,
                        max_lighting: float = 0.2, max_warp: float = 0.2, p_affine: float = 0.75,
                        p_lighting: float = 0.75, xtra_tfms=None, **kwargs):
    return get_transforms(do_flip, flip_vert, max_rotate, max_zoom, max_lighting, max_warp, p_affine, p_lighting,
                          xtra_tfms)


def base_arch_str_to_obj(base_arch):
    base_arch = models.resnet152 if base_arch in ['resnet152'] \
        else models.resnet101 if base_arch in ['resnet101'] \
        else models.resnet50 if base_arch in ['resnet50'] \
        else models.vgg16_bn if base_arch in ['vgg16', 'vgg16_bn'] or base_arch is None \
        else base_arch  # custom TODO not need create base_model in below line
    return base_arch


def viz_grid_images(files, is_show=False, png_file=None):
    """
    :param files: [rowxcol], must same size
    :param is_show: to show
    :param png_file: if not None then save
    :return:
    """
    all_imgs = []
    for row_f in files:
        row_imgs = [open_image(f).data for f in row_f]
        row_imgs = torch.stack(row_imgs, dim=0)
        all_imgs.append(row_imgs)

    batch_tensor = torch.cat(all_imgs, dim=0)

    # make with the number of column is the len of first row (all row must be same size)
    grid_img = torchvision.utils.make_grid(batch_tensor, nrow=len(files[0]))

    fig, ax = plt.subplots()
    ax.imshow(grid_img.permute(1, 2, 0))
    fig.savefig(png_file) if png_file is not None else None
    plt.show() if is_show else None

    return fig, ax


top2_acc = partial(top_k_accuracy, k=2)
top3_acc = partial(top_k_accuracy, k=3)
top5_acc = partial(top_k_accuracy, k=5)
top2_acc.__name__ = 'top2_accuracy'
top3_acc.__name__ = 'top3_accuracy'
top5_acc.__name__ = 'top5_accuracy'
tmetrics = [accuracy, top2_acc, top3_acc]

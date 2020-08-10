from builtins import super, Exception
from functools import partial
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from fastai import callbacks
from fastai.basic_data import DatasetType
from fastai.basic_train import Learner
from fastai.callback import Callback
from fastai.callbacks import SaveModelCallback, CSVLogger
from fastai.metrics import top_k_accuracy, accuracy, dice
from fastai.torch_core import MetricsList
from fastai.train import ClassificationInterpretation
from fastai.vision import imagenet_stats, get_transforms, models, open_image
from torch.autograd import Variable
from torch.nn.functional import log_softmax

from outside.scikit.plot_confusion_matrix import plot_confusion_matrix
from prlab.gutils import convert_to_obj, make_check_point_folder, convert_to_obj_or_fn, load_func_by_name, \
    lazy_object_fn_call
from prlab.torch.functions import ExLoss, weights_branches


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

    def on_epoch_end(self, epoch: int, smooth_loss: torch.Tensor, last_metrics: MetricsList, **kwargs: Any) -> bool:
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

    def on_epoch_end(self, epoch: int, smooth_loss: torch.Tensor, last_metrics: MetricsList, **kwargs: Any) -> bool:
        # do with data
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


# Loss functions/class
def do_reduction(loss_tensor, reduction='mean'):
    """
    reduction that use in most loss function
    :param loss_tensor:
    :param reduction: str => none (None), mean, sum
    :return: out with reduction
    """
    if isinstance(reduction, str) and hasattr(loss_tensor, reduction):
        loss_tensor = getattr(loss_tensor, reduction)()
    return loss_tensor


def dificult_weight_loss(pred, target, *args, **kwargs):
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


def prob_loss_raw(pred, target, **kwargs):
    """
    This is a custom of `torch.nn.CrossEntropyLoss`.
    NOTE: this version is nearly KL-divergence than CrossEntropy implement in pytorch?
    Loss when y input as raw probability instead a int number.
    Use when prob not 1/0 but float distribution
    :param pred: raw_scrore (not softmax)
    :param target: [0.7 0.2 0.1]
    :return:
    """
    # y = y[:, :7]
    l_softmax = log_softmax(pred, 1)

    a = -target * l_softmax
    a = torch.sum(a, dim=1)

    return a


class ProbLoss(torch.nn.Module):
    def __init__(self, reduction='mean', **kwargs):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        out = prob_loss_raw(pred=pred, target=target)
        return do_reduction(out, reduction=self.reduction)


class ProbLossRaw(ProbLoss):
    def __init__(self, **kwargs):
        kwargs['reduction'] = 'none'
        super().__init__(**kwargs)


def prob_loss(input, target, **kwargs):
    return torch.mean(prob_loss_raw(input, target, **kwargs))


class LabelSmoothingLoss(torch.nn.Module):
    """
    For Label Smoothing loss (not one-hot but softer) with alpha default 0.9.
    Similar to `prlab.fastai.utils.LabelSmoothingLoss1` but fix the hs of 1-alpha part and safer i_mtx
    pred: [bs, lbl_size], target: [bs]
    """

    def __init__(self, alpha=0.9, reduction='mean', **kwargs):
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target):
        lbl_size = pred.size()[-1]
        i_mtx = torch.eye(lbl_size).to(target.device)
        t = torch.embedding(i_mtx, target)
        soft_part = torch.ones_like(t)
        l_softmax = log_softmax(pred, 1)
        out = -t * l_softmax * self.alpha - soft_part * l_softmax * (1 - self.alpha) / lbl_size
        out = out.sum(dim=-1)

        return do_reduction(out, reduction=self.reduction)


class LabelSmoothingLoss1(torch.nn.Module):
    """
    For Label Smoothing loss (not one-hot but softer) with alpha default 0.9.
    Similar to `torch.nn.CrossEntropyLoss` but now with softer
    pred: [bs, lbl_size], target: [bs]
    NOTE:   there is a risk when does not sum out before the reduction step.
            the class mostly use with mean reduction, but if none given, then it will fail
            we do not correct it, just use newer version `LabelSmoothingLoss`.
            This class just keep for old reference.
    """

    def __init__(self, alpha=0.9, reduction='mean', **kwargs):
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.emb = None

    def forward(self, pred, target):
        if self.emb is None:
            self.emb = torch.eye(pred.size()[-1]).to(target.device)
        t = torch.embedding(self.emb, target)
        soft_part = torch.ones_like(t)
        l_softmax = log_softmax(pred, 1)
        out = -t * l_softmax * self.alpha - soft_part * l_softmax * (1 - self.alpha)

        return do_reduction(out, reduction=self.reduction)


SoftLoss = LabelSmoothingLoss1  # for old reference


class LabelSmoothingLossDis(LabelSmoothingLoss):
    """
    Like `LabelSmoothingLoss`, but the given is distribution (by statistical)
    pred: [bs, lbl_size], target: [bs, lbl_size]
    """

    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(alpha=alpha, **kwargs)

    def forward(self, pred, target):
        lbl_size = pred.size()[-1]
        i_mtx = torch.eye(lbl_size).to(target.device)

        with torch.no_grad():
            lbl_correct = target.argmax(dim=-1)
            one_hot = torch.embedding(i_mtx, lbl_correct)

        l_softmax = log_softmax(pred, 1)
        out = -(one_hot * self.alpha + target * (1 - self.alpha)) * l_softmax
        out = out.sum(dim=-1)

        return do_reduction(out, reduction=self.reduction)


class LabelSmoothingLossDisRaw(LabelSmoothingLossDis):
    def __init__(self, **kwargs):
        kwargs['reduction'] = 'none'  # just set none to return raw
        super().__init__(**kwargs)


class DistCutOffLoss(torch.nn.Module):
    """
    Use cross entropy loss but with the prior distribution and cut off for correct label
    ref to `torch.nn.CrossEntropyLoss`
    input pred [n_label] x target [n_label] (target is not in one-hot, it in form of distribution)
    lbl_correct = correct label = argmax(target)
    if pred[lbl_correct] >= target[lbl_correct] => loss = 0
    else: loss = cel(pred, target)
    reduction most of time is mean, sometime none if use inside of another loss that need raw.
    See `torch.nn.modules.loss._Loss`
    """

    def __init__(self, reduction='mean', part=2, **kwargs):
        super().__init__()
        self.reduction = reduction
        self.part = part

    def forward(self, pred, target):
        # calc cross entropy in raw mode (return [bs])
        out = prob_loss_raw(pred=pred, target=target)

        # calc weight, 0 or 1 only, no grad
        with torch.no_grad():
            lbl_correct = target.argmax(dim=-1)
            c_pred, c_target = pred[:, lbl_correct], target[:, lbl_correct]
            zeros, ones = torch.zeros(1), torch.ones(1)
            zeros, ones = zeros.to(target.device), ones.to(target.device)
            one_part = ones / self.part
            # [bs] which 0 or 1
            weighted = torch.where(c_pred < c_target, ones, one_part)
        out = out * weighted

        return do_reduction(out, reduction=self.reduction)


class DistCutOffLossRaw(DistCutOffLoss):
    def __init__(self, **kwargs):
        kwargs['reduction'] = 'none'  # just set none to return raw
        super().__init__(**kwargs)


class NormWeightsLoss(ExLoss):
    """
    Similar to `prlab.model.pyramid_sr.norm_weights_loss` but could configure the second loss func
    Note that the second loss func/obj must in raw (not reduction)
    """

    def __init__(self, second_loss_fn=None, **kwargs):
        print('run in NormWeightsLoss')
        super().__init__()
        self.f_loss = torch.nn.CrossEntropyLoss(reduction='none')  # default
        # TODO fix if second_loss_fn not support kwargs
        if second_loss_fn:
            self.f_loss = convert_to_obj_or_fn(second_loss_fn, **kwargs)
            print('use second loss', self.f_loss)

    def forward(self, pred, target, **kwargs):
        if not isinstance(pred, tuple):
            return self.f_loss(pred, target)

        out, _ = pred
        n_branches = out.size()[-1]
        losses = [self.f_loss(out[:, :, i], target) for i in range(n_branches)]

        losses = torch.stack(losses, dim=-1)
        # loss = (losses * weights).sum(dim=-1)

        loss = torch.mean(losses, dim=-1)
        return torch.mean(loss)


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


class NormWeightsAcc:
    """
    refer to `prlab.torch.functions.norm_weights_acc` but in class mode and could modified f_acc
    """
    __name__ = 'NormWeightsAcc'

    def __init__(self, part=-1, **config):
        second_metrics_fn = config.get('second_metrics_fn', 'fastai.metrics.accuracy')
        self.part = part
        self.f_acc, _ = load_func_by_name(second_metrics_fn)

    def __call__(self, pred, target, *args, **kwargs):
        c_out = weights_branches(pred=pred)
        if self.part >= 0:
            return self.f_acc(c_out[:, self.part], target[:, self.part])
        return self.f_acc(c_out, target)


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
    """
    Widely used for basic configure for fastai train/test
    :param config:
    :return:
    """
    config['path'] = Path(config['path'])
    config['model_path'] = Path(config['model_path'])

    cp, best_name, csv_log = make_check_point_folder(config, None, config.get('run', 'test'))
    loss_func = config.get('loss_func', None)
    config.update({
        'data_helper': convert_to_obj(config.get('data_helper', None), **config),
        'metrics': convert_to_obj_or_fn(config.get('metrics', None), **config),
        'loss_func': convert_to_obj_or_fn(loss_func, **config) if isinstance(loss_func, str) else loss_func,
        'tfms': get_transforms_wrap(xtra_tfms=[], **config),
        'callback_fn': lambda: get_callbacks(best_name=best_name, csv_filename=csv_log),
        'cp': cp,
    })
    print(cp)

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

iou = partial(dice, iou=True)
iou.__name__ = 'iou'

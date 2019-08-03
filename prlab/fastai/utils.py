from builtins import super, Exception
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from fastai import callbacks
from fastai.basic_data import DatasetType
from fastai.callbacks import SaveModelCallback, partial, CSVLogger, Learner, Tensor, MetricsList, Callback
from fastai.metrics import top_k_accuracy, accuracy
from fastai.train import ClassificationInterpretation
from fastai.vision import imagenet_stats
from torch.autograd import Variable
from torch.nn.functional import log_softmax


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

    def __init__(self, learn: Learner, monitor: str = 'accuracy', mode: str = 'max', every: str = 'improvement',
                 name: str = 'bestmodel_tta', step: int = 3, scale=1.1, monitor_func=accuracy, is_mul=True):
        super().__init__(learn, monitor=monitor, mode=mode, every=every, name=name)
        self.step = step
        self.scale = scale
        self.monitor_func = monitor_func
        self.is_mul = is_mul

    def on_epoch_end(self, epoch: int, **kwargs: Any) -> None:
        """Compare the value monitored to its best score and maybe save the model."""

        if epoch % self.step == 0:
            if self.every == "epoch":
                self.learn.save(f'{self.name}_{epoch}')
            else:  # every="improvement"
                val, tta_val, *_ = test_image_summary(self.learn, scale=self.scale, monitor_func=self.monitor_func,
                                                      k_tta=1,
                                                      is_plt=False)
                current = np.max(tta_val)

                if current is not None and self.operator(current, self.best):
                    print(f'Better model found at epoch {epoch} with {self.monitor} value: {current}.')
                    self.best = current
                    self.learn.save(f'{self.name}_{epoch}_{current}') if self.is_mul else \
                        self.learn.save(f'{self.name}')


class ECSVLogger(CSVLogger):
    """Extend CSVLogger, flush each epoch to easy track"""

    def __init__(self, learn: Learner, filename: str = 'history', append: bool = False):
        super(ECSVLogger, self).__init__(learn, filename, append)

    def on_epoch_end(self, epoch: int, smooth_loss: Tensor, last_metrics: MetricsList, **kwargs: Any) -> bool:
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

    def on_epoch_end(self, epoch: int, smooth_loss: Tensor, last_metrics: MetricsList, **kwargs: Any) -> bool:
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


def dificult_weight_loss(y_preds, target, *args, **kwargs):
    """
    loss = - 1 * log(pi) * (pj/pi), where pj is max(p_)
    :param y_preds:
    :param target:
    :param args:
    :param kwargs:
    :return:
    """
    softmax = log_softmax(y_preds, 1)
    target_one_hot = torch.zeros(len(target.cpu()), 7).scatter_(1, target.cpu().unsqueeze(1), 1.)

    xa, _ = torch.max(softmax, dim=1)

    a = -target_one_hot.cuda() * softmax
    a = torch.sum(a, dim=1)

    xaa = 1 + xa + a
    a = a * xaa
    return torch.mean(a)


def prob_loss(target, y, **kwargs):
    """
    This is a custom of SoftMaxCrossEntropy
    Loss when y input as raw probability instead a int number.
    Use when prob not 1/0 but float distribution
    :param target: raw_scrore (not softmax)
    :param y: [0.7 0.2 0.1]
    :return:
    """
    y = y[:, :7]
    l_softmax = log_softmax(target, 1)

    a = -y * l_softmax
    a = torch.sum(a, dim=1)

    return torch.mean(a)


def prob_acc(target, y, **kwargs):
    """
    accuracy when y input as raw probability instead a int number
    use when prob is not 1/0
    :param target: raw_score
    :param y: [0.7 0.2 0.1]
    :return: accuracy
    """
    return accuracy(target, torch.argmax(y, dim=1))


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


top2_acc = partial(top_k_accuracy, k=2)
top3_acc = partial(top_k_accuracy, k=3)
top2_acc.__name__ = 'top2_accuracy'
top3_acc.__name__ = 'top3_accuracy'
tmetrics = [accuracy, top2_acc, top3_acc]

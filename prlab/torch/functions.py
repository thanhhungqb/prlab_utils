from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from prlab.common.utils import load_func_by_name, convert_to_obj_or_fn, get_name


def cat_fixed(args):
    """ fix torch.cat when ele is zero-dimensional tensor """
    if len(args) > 0 and len(args[0].size()) == 0:
        args = [o.view(-1) for o in args]
    return torch.cat(args)


def cat_rec(args):
    """
    extend torch.cat that args should be tensor, list of tensor or list of list of tensor (rec)
    :param args: tensor, list of tensor or rec list of tensor. Should be same size to cat
    :return: tensor or list of tensor (rec) same structure as input
    """
    if isinstance(args, torch.Tensor):
        return cat_fixed(args)
    # list
    if isinstance(args[0], torch.Tensor):
        return cat_fixed(args)
    # list of list then do cat in each column separated and return same structure
    arrs = list(zip(*args))
    ret = [cat_rec(o) for o in arrs]
    return ret


class PassThrough(nn.Module):
    """
    Do nothing, just passthrough input
    """

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, *input, **kwargs):
        return input[0]


class AdaptiveConcatPool3d(nn.Module):
    """Layer that concats `AdaptiveAvgPool3d` and `AdaptiveMaxPool3d`.
    credit: https://www.kaggle.com/guntherthepenguin/fastai-v1-group-equivariate-cnns
    """

    def __init__(self, sz: Optional[int] = None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        sz = sz or 1
        self.ap, self.mp = nn.AdaptiveAvgPool3d(sz), nn.AdaptiveMaxPool3d(sz)

    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


class ExLoss(nn.Module):
    """
    Just extend to allow any params
    """

    def __init__(self, **kwargs):
        super().__init__()


class WrapLoss(nn.Module):
    """
    To wrap some loss does not support **kwargs
    from `torch.nn.modules.loss._Loss`
    """
    _wrap = None

    def __init__(self, **kwargs):
        super().__init__()
        if self._wrap is not None:
            self._obj = self._wrap(reduction=kwargs.get('reduction', 'mean'))

    def __call__(self, pred, target, *args, **kwargs):
        if self._obj is not None:
            return self._obj(pred, target, **kwargs)


class MSELossE(WrapLoss):
    _wrap = nn.MSELoss


class CrossEntropyLossE(nn.CrossEntropyLoss):
    def __init__(self, size_average=None,
                 ignore_index: int = -100,
                 reduction: str = 'mean', **kwargs):
        super(CrossEntropyLossE, self).__init__(size_average=size_average, ignore_index=ignore_index,
                                                reduction=reduction)

    def forward(self, input, target, **kw):
        return super().forward(input, target)


def prob_weights_loss(pred, target, **kwargs):
    """
    `CrossEntropyLoss` but for multi branches (out, weight) :([bs, C, branches], [bs, branches])
    :param pred:
    :param target:
    :param kwargs:
    :return:
    """
    f_loss, _ = load_func_by_name('prlab.fastai.utils.prob_loss_raw')
    if not isinstance(pred, tuple):
        return f_loss(pred, target)

    out, _ = pred
    n_branches = out.size()[-1]
    losses = [f_loss(out[:, :, i], target) for i in range(n_branches)]

    losses = torch.stack(losses, dim=-1)
    # loss = (losses * weights).sum(dim=-1)
    loss = torch.mean(losses, dim=-1)
    return torch.mean(loss)


def weights_branches(pred, **kwargs):
    """
    Use together with `prlab.model.pyramid_sr.prob_weights_loss`
    :param pred: out, weights: [bs, n_classes, n_branches] [bs, n_branches]
    :param kwargs:
    :return:
    """
    if not isinstance(pred, tuple):
        # if not tuple then return the final/weighted version => DO NOTHING
        return pred

    out, _ = pred

    n_branches = out.size()[-1]
    sm = [torch.softmax(out[:, :, i], dim=1) for i in range(n_branches)]
    sm = torch.stack(sm, dim=-1)
    # c_out = torch.bmm(sm, weights.unsqueeze(-1)).squeeze(dim=-1)
    c_out = torch.mean(sm, dim=-1)

    return c_out


# ====================== METRICS ========================
class CompoundMetric:
    """
    one or more loss function in single tensor. e.g. [bs, 5] with reg in [bs, 0:1] and cls in [bs, 1:]
    """

    def __init__(self, metric_fns, cut_points, ret_only=None, **kwargs):
        """
        :param metric_fns: metric function/class callable
        :param cut_points: cut_point in second of input [bs, ?, ...], len(cut_point) = len(metric_fns)-1
        :param ret_only: int or None, what metric (only one) should be return, None if return all
        """
        super().__init__()
        assert len(metric_fns) - 1 == len(cut_points)
        self.metric_fns = [convert_to_obj_or_fn(o) for o in metric_fns]
        self.cut_points = [0] + cut_points + [None]  # to easy to for loop with first and last of list
        self.ret_only = ret_only

        if ret_only is not None: self.__name__ = get_name(self.metric_fns[ret_only])

    def __repr__(self):
        return f"CompoundMetric ({[get_name(o) for o in self.metric_fns]})"

    def __call__(self, pred, target, **kwargs):
        if self.ret_only is not None:
            metric = self.metric_fns[self.ret_only]
            s, e = self.cut_points[self.ret_only], self.cut_points[self.ret_only + 1]
            val = metric(pred[:, s:e], target[:, s:e])
            return val

        each_metric = []
        for i in range(len(self.metric_fns)):
            s, e = self.cut_points[i], self.cut_points[i + 1]
            print('test', s, e)
            print('val', pred[:, s:e], target[:, s:e])
            val = self.metric_fns[i](pred[:, s:e], target[:, s:e])
            each_metric.append(val)

        metrics = torch.stack(each_metric, dim=0)
        return metrics


def prob_weights_acc(pred, target, **kwargs):
    """
    Use together with `prlab.model.pyramid_sr.prob_weights_loss`
    :param pred: out, weights: [bs, n_classes, n_branches] [bs, n_branches]
    :param target: int for one-hot and list of float for prob
    :param kwargs:
    :return:
    """
    f_acc, _ = load_func_by_name('prlab.fastai.utils.prob_acc')
    c_out = weights_branches(pred=pred)

    return f_acc(c_out, target)  # f_acc(pred[0][:, :, 0], target)


def norm_weights_loss(pred, target, **kwargs):
    """
    `CrossEntropyLoss` but for multi branches (out, weight) :([bs, C, branches], [bs, branches])
    :param pred:
    :param target:
    :param kwargs:
    :return:
    """
    # f_loss, _ = load_func_by_name('prlab.fastai.utils.prob_loss_raw')
    f_loss = nn.CrossEntropyLoss()
    if not isinstance(pred, tuple):
        return f_loss(pred, target)

    out, _ = pred
    n_branches = out.size()[-1]
    losses = [f_loss(out[:, :, i], target) for i in range(n_branches)]

    losses = torch.stack(losses, dim=-1)
    # loss = (losses * weights).sum(dim=-1)
    loss = torch.mean(losses, dim=-1)
    return torch.mean(loss)


def norm_weights_acc(pred, target, **kwargs):
    """
    Use together with `prlab.model.pyramid_sr.prob_weights_loss`
    :param pred: out, weights: [bs, n_classes, n_branches] [bs, n_branches]
    :param target: int for one-hot and list of float for prob
    :param kwargs:
    :return:
    """
    f_acc, _ = load_func_by_name('fastai.metrics.accuracy')
    c_out = weights_branches(pred=pred)

    return f_acc(c_out, target)  # f_acc(pred[0][:, :, 0], target)


class WeightsAcc:
    """
    More general than `norm_weights_acc` and `prob_weights_acc`.
    Can set the base function
    """
    __name__ = 'WeightsAcc'

    def __init__(self, base_acc=None, **kwargs):
        super().__init__()

        default_name = 'fastai.metrics.accuracy'
        acc_default_fn, _ = load_func_by_name(default_name)
        if base_acc is None:
            self.base_acc = acc_default_fn
        else:
            self.base_acc = load_func_by_name(base_acc)[0] \
                if isinstance(base_acc, str) else base_acc

        tmp = base_acc if base_acc is not None else default_name
        self.__name__ = 'WeightsAcc ({})'.format(tmp)

    def __call__(self, pred, target, **kwargs):
        c_out = weights_branches(pred=pred)

        return self.base_acc(c_out, target)


def mae(pred, target, **kwargs):
    return torch.abs(target - pred).mean()


def concordance_index(pred, target, event_observed=None, **kwargs):
    from lifelines.utils import concordance_index
    ret = concordance_index(target.detach().cpu(), pred.detach().cpu(), event_observed=event_observed)
    ret = torch.tensor(ret)
    return ret


# ====================== END OF METRICS ========================

def make_theta_from_st(st, is_inverse=False):
    """
    ST without rotate
    note: batch mode [-1, 4] to [-1, 2, 3]
    e[1], e[3] in ratio, scale mode (mean not need *2/w or *2/h)

    ref: https://discuss.pytorch.org/t/how-to-convert-an-affine-transform-matrix-into-theta-to-use-torch-nn-functional-affine-grid/24315/3

    theta[0,0] = param[0,0]
    theta[0,1] = param[0,1]*h/w
    theta[0,2] = param[0,2]*2/w + theta[0,0] + theta[0,1] - 1
    theta[1,0] = param[1,0]*w/h
    theta[1,1] = param[1,1]
    theta[1,2] = param[1,2]*2/h + theta[1,0] + theta[1,1] - 1

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
    theta = torch.stack(o, dim=-1)

    theta[:, 2] = theta[:, 2] + theta[:, 0] + theta[:, 1] - 1
    theta[:, 5] = theta[:, 5] + theta[:, 3] + theta[:, 4] - 1

    return theta.view(-1, 2, 3)


def soft_argmax(x=None):
    """
    Implement soft-argmax, that can differentiable.
    ```
        soft_argmax(torch.Tensor([[1.1, 3.0, 1.1, 1.3, 0.8]]))
    ```
    implement in batch size
    ref: https://medium.com/@nicolas.ugrinovic.k/soft-argmax-soft-argmin-and-other-soft-stuff-7f94e6120dff
    :param x: [-1, size], if x more than 2, then convert to 2 and calc argmax, and then convert back (by row, col)
    :return: max[-1], argmax[-1]
    """
    beta = 12
    a = torch.exp(beta * x)
    b = torch.sum(torch.exp(beta * x), dim=-1)
    soft_max = a / b.unsqueeze(dim=-1)
    max_val = torch.sum(soft_max * x, dim=-1)
    pos = torch.arange(0, x.size()[-1])
    s_argmax = torch.sum(soft_max * pos, dim=-1)
    return max_val, s_argmax


def build_grid(source_size, target_size):
    """
    ref: https://discuss.pytorch.org/t/cropping-a-minibatch-of-images-each-image-a-bit-differently/12247/5
    if w != h then need change small thing
    :param source_size:
    :param target_size:
    :return:
    """
    k = float(target_size) / float(source_size)
    direct = torch.linspace(-k, k, target_size).unsqueeze(0).repeat(target_size, 1).unsqueeze(-1)
    full = torch.cat([direct, direct.transpose(1, 0)], dim=2).unsqueeze(0)
    return full


def random_crop_grid(x, grid):
    delta = x.size(2) - grid.size(1)
    grid = grid.repeat(x.size(0), 1, 1, 1)
    # Add random shifts by x
    grid[:, :, :, 0] = grid[:, :, :, 0] + \
                       torch.FloatTensor(x.size(0)).random_(0, delta) \
                           .unsqueeze(-1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2)) / x.size(2)
    # Add random shifts by y
    grid[:, :, :, 1] = grid[:, :, :, 1] + \
                       torch.FloatTensor(x.size(0)).random_(0, delta) \
                           .unsqueeze(-1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2)) / x.size(2)
    return grid


def fc_more_label(fc, n_label=1):
    """
    extend some row (more labels) for fc in pytorch
    :param fc:
    :param n_label: number of labels to add, default is 1
    :return:
    """
    o_label = fc.weight.size()[0]
    n_label_size = o_label + n_label

    new_fc = nn.Linear(fc.weight.size()[1], n_label_size)

    w = fc.weight  # size label, in_size
    b = fc.bias  # size label

    # new w and b is (new_label_size, in_size) and (new_label_size)
    new_fc.weight.data[:, :] = F.pad(w, (0, 0, 0, 1), 'constant', 0)
    new_fc.bias.data[:] = F.pad(b, (0, 1), "constant", 0)

    return new_fc


def fc_cut_label(fc, n_label=1, in_place=False):
    """
    cut some row (less labels) for fc in pytorch. Note: careful with the order, will cut the latest
    If the order not correct, call `fc_exchange_label` before call this function
    :param fc:
    :param n_label: number of labels to remove, default is 1
    :param in_place:
    :return:
    """
    o_label = fc.weight.size()[0]
    n_label_size = o_label - n_label

    new_fc = nn.Linear(fc.weight.size()[1], n_label_size)

    w, b = fc.weight, fc.bias

    new_fc.weight.data.copy_(w[:n_label_size, :])
    new_fc.bias.data.copy_(b[:n_label_size])

    if in_place:
        fc.weight, fc.bias = new_fc.weight, new_fc.bias
        return fc
    else:
        return new_fc


def fc_exchange_label(fc, new_pos=None, in_place=False):
    """
    exchange some label order with new_pos given
    the new labels size may be differ from old, in case greater, try to fill new_pos with some
    repeat to make sure the size, [0, 2, 1, 4, 3, 0, 0, 0...]
    :param fc:
    :param new_pos:
    :param in_place: if do in current fc
    :return:
    """
    if new_pos is None:  # keep current order, do nothing
        return fc

    new_fc = nn.Linear(fc.weight.size()[1], len(new_pos))

    # reorder step
    w = [fc.weight[pos] for pos in new_pos]
    b = [fc.bias[pos] for pos in new_pos]
    w, b = torch.stack(w, dim=0), torch.stack(b)

    if in_place:
        # fc.weight.data.copy_(w), fc.bias.data.copy_(b)
        new_fc.weight.data.copy_(w), new_fc.bias.data.copy_(b)
        fc.weight, fc.bias = new_fc.weight, new_fc.bias
        return fc
    else:
        new_fc.weight.data.copy_(w), new_fc.bias.data.copy_(b)
        return new_fc


# # We want to crop a 80x80 image randomly for our batch
# # Building central crop of 80 pixel size
# grid_source = build_grid(batch.size(2), 80)
# # Make radom shift for each batch
# grid_shifted = random_crop_grid(batch, grid_source)
# # Sample using grid sample
# sampled_batch = F.grid_sample(batch, grid_shifted)

class TransformsWrapFn:
    # transform wrap from fn(img_tensor, ...) -> img_tensor
    # Usage:
    #       TransformsWrapFn(crop_3d_volume, fixed_params=['crop_dim', 'crop_size'], crop_dim=256, crop_size=256)
    #       TransformsWrapFn(random_zoom, fixed_params=['min_percentage', 'max_percentage'],
    #           min_percentage=0.7, max_percentage=1.2)
    def __init__(self, fn, fixed_params=None, **params):
        """
        :param fn: callable
        :param fixed_params: list of param names
        :param params: param to pass to fn
        """
        self.fn = convert_to_obj_or_fn(fn)
        self.params = params
        self.fixed_params = fixed_params

    def __call__(self, *args, **kwargs):
        params = {**kwargs, **self.params}
        if self.fixed_params is not None:
            params = {k: params[k] for k in self.fixed_params}
        return self.fn(*args, **params)

    def __str__(self):
        return f"prlab.torch.functions.TransformsWrapFn ( {str(self.fn)} )"

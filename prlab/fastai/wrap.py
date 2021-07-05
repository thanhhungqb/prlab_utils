"""
Wrap exist class, function in fastai for some special task
"""

import torch
import torch.nn as nn
from fastai.tabular.model import TabularModel
from fastcore.basics import listify, ifnone


def get_tabular_model(data_train, layers,
                      out_sz=None,
                      emb_szs=None,
                      ps=None, emb_drop: float = 0., y_range=None, use_bn: bool = True,
                      **kwargs):
    """
    Wrap `fastai.tabular.learner.tabular_learner` to get model only
    TODO upgrade to fastai v2
    data_train:DataBunch, emb_szs: Dict[str, int], y_range: OptRange
    layers: Collection[int], ps: Collection[float]
    """
    emb_szs = data_train.get_emb_szs(ifnone(emb_szs, {}))
    model = TabularModelEx(emb_szs, len(data_train.cont_names),
                           out_sz=data_train.c if out_sz is None else out_sz,
                           layers=layers, ps=ps,
                           emb_drop=emb_drop, y_range=y_range, use_bn=use_bn, **kwargs)

    return model


class TabularModelEx(TabularModel):
    """
    Extended the Basic model for tabular data from `fastai.tabular.models.TabularModel`
    If is_only_output then the class equals to original version, else return the embedded_x to use outside
    """

    def __init__(self, emb_szs, n_cont, out_sz, layers, ps=None, embed_p=0.,
                 y_range=None, use_bn=True, bn_final=False, bn_cont=True, act_cls=nn.ReLU(inplace=True),
                 is_only_output=True,
                 **kw):
        super(TabularModelEx, self).__init__(
            emb_szs=emb_szs, n_cont=n_cont, out_sz=out_sz, layers=layers, ps=ps, embed_p=embed_p,
            y_range=y_range, use_bn=use_bn, bn_final=bn_final, bn_cont=bn_cont, act_cls=act_cls
        )
        self.is_only_output = is_only_output

    def forward(self, x_cat, x_cont=None, **kw):
        if isinstance(x_cat, list):
            # pass cat and cont in form of list 2 elements
            x_cat, x_cont, *_ = x_cat

        x_p = x_cont  # x_p keep original input vector after embedding without dropout
        if self.n_emb != 0:
            x = [e(x_cat[:, i]) for i, e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x_p = x
            x = self.emb_drop(x)
        if self.n_cont != 0:
            if self.bn_cont is not None: x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
            x_p = torch.cat([x_p, x_cont], dim=-1)

        out = self.layers(x)
        return out if self.is_only_output else (out, x_p)


class SimpleDNN(nn.Module):
    """
    Very simple DNN model with custom the deep, number nodes in each layer.
    """

    def __init__(self, input_size, hidden_size, n_classes,
                 is_relu=True, dropout=[None],
                 use_bn=True, bn_final=False,
                 **kwargs):
        """
        :param input_size: the input dimension
        :param hidden_size: int or [int], nodes in hidden layer(s)
        :param n_classes: the output dimension
        :param is_relu: if use relu after each hidden layer
        """
        super(SimpleDNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes

        self.hidden_size = self.hidden_size if isinstance(self.hidden_size, list) else [self.hidden_size]
        if len(self.hidden_size) == 0: raise Exception("hidden_size should be not empty list")

        self.is_relu = listify(is_relu, self.hidden_size)
        self.dropout = listify(dropout, self.hidden_size)
        self.use_bn = listify(use_bn, self.hidden_size)
        self.bn_final = bn_final

        seq = []
        prev = self.input_size
        # seq.append(nn.ReLU()) if self.is_relu else None
        # layers.append(nn.BatchNorm1d(self.hidden_size[0])) if self.use_bn else None

        for pos, (n_nodes, is_relu, is_bn, drop, *_) in enumerate(
                zip(self.hidden_size, self.is_relu, self.use_bn, self.dropout)):
            seq.append(nn.Linear(prev, n_nodes))
            seq.append(nn.ReLU()) if is_relu else None
            seq.append(nn.BatchNorm1d(n_nodes)) if is_bn else None
            seq.append(nn.Dropout(p=drop)) if drop is not None else None

            prev = n_nodes

        seq.append(nn.Linear(self.hidden_size[-1], self.n_classes))
        seq.append(nn.BatchNorm1d(self.n_classes)) if self.bn_final else None

        self.seq = nn.Sequential(*seq)

    def forward(self, *x):
        return self.seq(*x)

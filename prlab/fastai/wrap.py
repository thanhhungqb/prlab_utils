"""
Wrap exist class, function in fastai for some special task
"""
from fastai.tabular import *


def get_tabular_model(data_train: DataBunch, layers: Collection[int],
                      out_sz=None,
                      emb_szs: Dict[str, int] = None,
                      ps: Collection[float] = None, emb_drop: float = 0., y_range: OptRange = None, use_bn: bool = True,
                      **kwargs):
    """Wrap `fastai.tabular.learner.tabular_learner` to get model only"""
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

    def __init__(self, emb_szs, n_cont: int, out_sz: int, layers: Collection[int],
                 ps: Collection[float] = None, emb_drop: float = 0., y_range: OptRange = None,
                 use_bn: bool = True, bn_final: bool = False,
                 is_only_output=True,
                 **kwargs):

        # support lazy calculate/make for emb_szs
        # see `fastai.tabular.learner.tabular_learner` and `fastai.tabular.models.TabularModel`, there is different of
        # meaning of emb_szs (ListSizes and Dict)
        if isinstance(emb_szs, dict):  # TODO how to check ListSizes
            # in this case, lazy calc is used, data_train should be here
            data_train = kwargs.get('data_train')
            emb_szs = data_train.get_emb_szs(ifnone(emb_szs, {}))

        super(TabularModelEx, self).__init__(
            emb_szs=emb_szs, n_cont=n_cont,
            out_sz=out_sz, layers=layers,
            ps=ps, emb_drop=emb_drop,
            y_range=y_range,
            use_bn=use_bn, bn_final=bn_final
        )

        self.is_only_output = is_only_output

    def forward(self, x_cat: Tensor, x_cont: Tensor) -> Tensor:
        if self.n_emb != 0:
            x = [e(x_cat[:, i]) for i, e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont

        embedded_x = x
        x = self.layers(x)
        if self.y_range is not None:
            x = (self.y_range[1] - self.y_range[0]) * torch.sigmoid(x) + self.y_range[0]
        return x if self.is_only_output else x, embedded_x

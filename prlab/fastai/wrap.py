"""
Wrap exist class, function in fastai for some special task
"""
from fastai.tabular import *


def get_tabular_model(data_train: DataBunch, layers: Collection[int],
                      out_sz=None,
                      emb_szs: Dict[str, int] = None,
                      ps: Collection[float] = None, emb_drop: float = 0., y_range: OptRange = None, use_bn: bool = True,
                      **learn_kwargs):
    """Wrap `fastai.tabular.learner.tabular_learner` to get model only"""
    emb_szs = data_train.get_emb_szs(ifnone(emb_szs, {}))
    model = TabularModel(emb_szs, len(data_train.cont_names),
                         out_sz=data_train.c if out_sz is None else out_sz,
                         layers=layers, ps=ps,
                         emb_drop=emb_drop, y_range=y_range, use_bn=use_bn)

    return model

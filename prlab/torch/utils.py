import logging

import numpy as np
import torch

from prlab.gutils import convert_to_obj_or_fn

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)


def process_input(i_features, i_targets, device='cuda', **config):
    """
    Process and convert to device (cpu, gpu).
    support multi-feature (array) and multi-target (array)
    :param i_features: numpy array or list of numpy array (int or float)
    :param i_targets: numpy array or list of numpy array
    :param device: cuda or cpu
    :param config:
    :return:
    """
    features = i_features.to(device) if isinstance(i_features, (np.ndarray, torch.Tensor)) else \
        [o.to(device) for o in i_features]
    targets = i_targets.to(device) if isinstance(i_targets, (np.ndarray, torch.Tensor)) else \
        [o.to(device) for o in i_targets]

    return features, targets


def process_metrics(preds_lst, targets_lst, **config):
    """
    TODO some case, when output is not single tensor, then this command will get error
    TODO work with tensor instead numpy as it directly output of model (maybe in gpu also)
    each element in list is a batch output, not single, then flatten is need before cal
    :param preds_lst: each element is preds for one BATCH
    :param targets_lst: each element is target for one BATCH
    :param config:
    :return:
    """

    metrics = [convert_to_obj_or_fn(o, **config) for o in config.get('metrics', [])]

    # TODO flatten BATCH mode

    targets, preds = np.array(targets_lst), np.array(preds_lst)
    metrics_score = np.array([metric(preds, targets) for metric in metrics])

    return metrics_score


def one_epoch(data_loader, model, loss_fn, optimizer, test_mode=False, **config):
    """
    Run single epoch in both train and test/valid
    :param data_loader:
    :param model:
    :param loss_fn:
    :param optimizer:
    :param config:
    :param test_mode
    :return:
    """
    running_loss = 0.0

    predictions_corr = list()
    labels_corr = list()

    model.train() if not test_mode else model.eval()
    with torch.no_grad() if test_mode else meaningless_ctx():
        for idx, (i_features, i_targets) in enumerate(data_loader):
            optimizer.zero_grad() if not test_mode else None

            features, targets = process_input(i_features=i_features, i_targets=i_targets, **config)
            predictions = model(features)
            loss = loss_fn(predictions, targets)

            (loss.backward(), optimizer.step()) if not test_mode else None

            running_loss += loss.item()
            predictions_corr += predictions
            labels_corr += targets

    metric_scores = process_metrics(preds_lst=predictions_corr, targets_lst=labels_corr, **config)
    return running_loss / len(labels_corr), metric_scores


class meaningless_ctx:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

import json
import logging
import time

import numpy as np
import torch

from prlab.common.utils import convert_to_obj_or_fn, to_json_writeable

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_control(**config):
    train_logger = config.get('train_logger', logger)
    progress_logger = config.get('progress_logger', train_logger)
    train_logger.info(json.dumps(to_json_writeable(config), indent=2))
    train_logger.info("** start training ...")
    train_logger.info(f"number of trainable parameters {count_parameters(config['model'])}")

    metric_names = [f'n_{i}' for i in range(5)]

    val_best_loss = None

    for epoch in range(int(config["epochs"])):
        start_time = time.time()

        # one epoch train and validate
        train_loss, train_score, *_ = one_epoch(**config)
        val_loss, val_score, *_ = one_epoch(test_mode=True, **config)

        if val_best_loss is None or val_loss < val_best_loss:
            # TODO best valid callback
            val_best_loss, val_best_score = val_loss, val_score
            torch.save(config['model'].state_dict(), config['cp'] / 'best.w')

        metrics = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            "time": time.time() - start_time,
        }
        metrics = {**metrics,
                   **{f'train_{metric_names[i]}': float(train_score[i]) for i in range(len(train_score))},
                   **{f'val_{metric_names[i]}': float(val_score[i]) for i in range(len(val_score))}}

        progress_logger.info(json.dumps(to_json_writeable(metrics)))

    # log the best valid
    msg1 = {
        "Best validation loss": float(val_best_loss),
        **{f'best_val_{metric_names[i]}': float(val_best_score[i]) for i in range(len(val_best_score))}
    }
    msg1 = to_json_writeable(msg1)
    train_logger.info(json.dumps(msg1))
    progress_logger.info(json.dumps(msg1))

    return config


def eval_control(**config):
    train_logger = config.get('train_logger', logger)
    progress_logger = config.get('progress_logger', train_logger)
    train_logger.info("** start testing ...")

    metric_names = [f'n_{i}' for i in range(5)]

    start_time = time.time()

    # eval in test set here
    to_test_config = {**config, 'data_valid': config['data_test']}
    test_loss, val_score, *_ = one_epoch(test_mode=True, **to_test_config)

    metrics = {'test_loss': float(test_loss),
               "time": "{:.2f}".format((time.time() - start_time)),
               **{f'test_{metric_names[i]}': float(val_score[i]) for i in range(len(val_score))}}

    msg1 = to_json_writeable(metrics)
    progress_logger.info(json.dumps(msg1))
    train_logger.info(json.dumps(msg1))

    return config


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


def process_metrics(preds_tensor, targets_tensor, **config):
    """
    TODO some case, when output is not single tensor, then this command will get error
    TODO work with tensor instead numpy as it directly output of model (maybe in gpu also)
    each element in list is a batch output, not single, then flatten is need before cal
    :param preds_tensor: each element is preds for one BATCH
    :param targets_tensor: each element is target for one BATCH
    :param config:
    :return:
    """
    metrics = [convert_to_obj_or_fn(o, **config) for o in config.get('metrics', [])]
    metrics_score = np.array([metric(preds_tensor, targets_tensor) for metric in metrics])

    return metrics_score


def one_epoch(model, loss_func, opt_func, data_loader=None, test_mode=False, **config):
    """
    Run single epoch in both train and test/valid
    :param data_loader:
    :param model:
    :param loss_func:
    :param opt_func:
    :param config:
    :param test_mode
    :return:
    """
    running_loss = 0.0

    predictions_corr = list()
    labels_corr = list()

    model.train() if not test_mode else model.eval()
    data_loader = data_loader if data_loader is not None else \
        (config['data_train'] if not test_mode else config.get('data_valid', config['data_test']))

    n_count = 0
    with torch.no_grad() if test_mode else meaningless_ctx():
        for idx, (i_features, i_targets) in enumerate(data_loader):
            opt_func.zero_grad() if not test_mode else None

            features, targets = process_input(i_features=i_features, i_targets=i_targets, **config)
            predictions = model(features)
            predictions = torch.squeeze(predictions, -1)

            loss = loss_func(predictions, targets)

            (loss.backward(), opt_func.step()) if not test_mode else None

            running_loss += loss.item()
            n_count += 1
            predictions_corr += predictions
            labels_corr += targets

    loss_val = running_loss / n_count
    predictions_corr, labels_corr = torch.tensor(predictions_corr), torch.tensor(labels_corr)
    metric_scores = process_metrics(preds_tensor=predictions_corr, targets_tensor=labels_corr, **config)
    return loss_val, metric_scores


class meaningless_ctx:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

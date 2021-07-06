"""
Implement many pipe function with form:
    fn(**config:dict) -> config(dict)
Class Pipe also supported:
    class A:
        def __call__(**config:dict) -> config(dict)
"""
import logging
from pathlib import Path

import torch

from prlab.common.utils import convert_to_obj_or_fn

logger = logging.getLogger(__name__)


def simple_model(**config):
    config['model'] = convert_to_obj_or_fn(config['model'], **config).to(config.get('device', 'cuda'))
    return config


def model_load_weights(**config):
    """
    Load model weights, should be call after model available
    Require:
        model
        strict (opt) to check parameters name
        cp (Path)
        loggers
    """
    model = config['model']
    model_weights_path = config.get('model_weights_path', config['cp'] / 'best.w')
    model_weights_path = Path(model_weights_path)

    if model_weights_path.is_file():
        train_logger = config.get('train_logger', logger)
        train_logger.info(f"try to load weights from {str(model_weights_path)}")

        out = model.load_state_dict(torch.load(model_weights_path), strict=config.get('strict', False))
        train_logger.info(f"load weights msg:\n {str(out)}")

    return config


# ================= ENF OF MODEL PART ==================

def opt_func_load(**config):
    # TODO fix opt load, maybe wrap to object to predefined params, such as lr, ...
    opt_fn = convert_to_obj_or_fn(config['opt_func'])
    config['opt_func'] = opt_fn(params=config['model'].parameters(), lr=config.get('lr'))
    return config


def model_general_setup(**config):
    """
    General setup after model build, including opt_func, loss_func, metrics
    :param config:
    :return:
    """
    config = opt_setup(**config)
    config['loss_func'] = convert_to_obj_or_fn(config['loss_func'], **config)
    config['metrics'] = convert_to_obj_or_fn(config['metrics'], **config)

    return config


def opt_setup(**config):
    """
    Setup for opt func, support single lr and per-parameter
    :param config:
    :return:
    """
    if callable(config['opt_func']):
        return config

    if isinstance(config['opt_func'], list):
        if isinstance(config['opt_func'][-1].get('lr', 1e-2), list):  # per-parameter lr
            # then model should has layer_groups
            assert hasattr(config['model'], 'layer_groups'), "model should has layer_groups"
            groups = config['model'].layer_groups()
            lrs = config['opt_func'][-1]['lr']
            assert len(groups) == len(lrs), "should same size"

            # make param by groups and its lr
            params = [{'params': groups[i].parameters(), 'lr': lrs[i]} for i in range(len(groups))]

            pass_params = {**config['opt_func'][-1], 'params': params, 'lr': lrs[-1]}
            config['opt_func'] = convert_to_obj_or_fn(config['opt_func'][:2] + [pass_params])
        else:  # should be single lr
            config['opt_func'] = convert_to_obj_or_fn(config['opt_func'], params=config['model'].parameters())
    else:  # str, etc.
        config['opt_func'] = convert_to_obj_or_fn(config['opt_func'], params=config['model'].parameters())

    return config

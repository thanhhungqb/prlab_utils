import logging
from pathlib import Path

from prlab.common.logger import PrettyLineHandler, WandbHandler
from prlab.gutils import make_check_point_folder, convert_to_obj_or_fn

logger = logging.getLogger(__name__)


# ========================  GENERAL ========================
def general_dl_make_up(**config):
    """
    Widely used for basic configure for deep learning train/test
    :param config:
    :return:
    """
    config['path'] = Path(config['path'])
    config['model_path'] = Path(config['model_path'])

    cp, best_name, *_ = make_check_point_folder(config, None, config.get('run', 'test'))
    loss_func = config.get('loss_func', None)
    config.update({
        'data_helper': convert_to_obj_or_fn(config.get('data_helper'), **config),
        'metrics': convert_to_obj_or_fn(config.get('metrics', []), **config),
        'loss_func': convert_to_obj_or_fn(loss_func, **config),
        'cp': cp,
    })

    return config


def make_train_loggers(**config):
    """ There are two logger including general logger to stdout/file and another to progress (stdout/wandb)"""
    name = config.get('run', 'general')
    proj_name = config.get('proj_name', name)
    log_level = config.get('log_level', logging.INFO)

    logger_general = logging.getLogger(name)
    logger_general.setLevel(log_level)
    logger_general.addHandler(logging.StreamHandler())
    logger_general.addHandler(logging.FileHandler(config["cp"] / "general.log"))

    logger_progress = logging.getLogger(f"{name}_progress")
    logger_progress.setLevel(log_level)
    logger_progress.addHandler(PrettyLineHandler())
    logger_progress.addHandler(PrettyLineHandler(base_hdl=logging.FileHandler(config["cp"] / "progress.log")))
    if config.get('wandb') is not None:
        logger_progress.addHandler(WandbHandler(proj_name=proj_name))

    return {**config, 'train_logger': logger_general, 'progress_logger': logger_progress}
# ========================  END OF GENERAL ========================

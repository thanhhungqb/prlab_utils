import logging
from pathlib import Path

from prlab.common.logger import PrettyLineHandler, WandbHandler
from prlab.common.utils import make_check_point_folder, convert_to_obj_or_fn

logger = logging.getLogger(__name__)


# ======================== CONTROL =========================
def pipeline_control_multi(**kwargs):
    """
    For general control overall pipeline, support for multi pipeline
    See `prlab.fastai.pipeline.pipeline_control`
    What different:
        - `prlab.fastai.utils.general_configure` add to pipe instead call directly
        - support many pipeline in order name (max1000): process_pipeline_0, process_pipeline_1, ...
        Why process_pipeline in the middle? FOR SUPPORT THE OLDER VERSION of configure file
    :param kwargs: configure
    :return:
    """
    config = {}
    config.update(**kwargs)

    ordered_pipeline_names = ['process_pipeline_{}'.format(i) for i in range(1000) if
                              config.get('process_pipeline_{}'.format(i), None) is not None]
    # sometime set 'none' instead list to disable it (override json by command line)
    ordered_pipeline_names = [o for o in ordered_pipeline_names if isinstance(config[o], list)]
    if len(ordered_pipeline_names) == 0:
        # support old version of configure file
        ordered_pipeline_names = ['process_pipeline']

    # TODO
    #   Do we need object create and call here, if then lazy should be consider for all pipe
    #   then local and global params should be careful consider
    #   for lazy load, another problem with module load at runtime also have to care because
    #   some case, source code are remove immediately after main process run
    # this step to make sure all name could be convert to function to call later
    # this is early check
    for pipe_name in ordered_pipeline_names:
        convert_to_obj_or_fn(config[pipe_name])

    for pipe_name in ordered_pipeline_names:
        # the previous output config may be affect the next step of the pipeline,
        # then convert_to_obj_or_fn should be call after previous step done.
        process_pipeline = convert_to_obj_or_fn(config[pipe_name])
        for fn in process_pipeline:
            fn = convert_to_obj_or_fn(fn, lazy=True)
            if callable(fn):
                config = fn(**config)
            else:
                logger = config.get('train_logger', logging.getLogger(__name__))
                logger.warning(f"{fn} is not callable")

    return config


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
        logger_progress.addHandler(WandbHandler(**config))

    return {**config, 'train_logger': logger_general, 'progress_logger': logger_progress}
# ========================  END OF GENERAL ========================

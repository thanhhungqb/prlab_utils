from pathlib import Path

from prlab.gutils import make_check_point_folder, convert_to_obj_or_fn


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

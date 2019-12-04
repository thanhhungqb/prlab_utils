import importlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing

# add ls function to Path for easy to use
Path.ls = lambda x: list(x.iterdir())


def to_json_writeable(js):
    """
    Convert js in JSON to JSON writeable to file, means only str, number, list
    :param js:
    :return:
    """
    if isinstance(js, (int, float)):
        return js
    if isinstance(js, str):
        return js
    if isinstance(js, (tuple, list)):
        return [to_json_writeable(o) for o in js]
    if isinstance(js, dict):
        return {k: to_json_writeable(v) for k, v in js.items()}

    # other wise, try to convert to str
    return str(js)


def make_check_point_folder(config={}, cp_base=None, cp_name=None):
    """
    make a checkpoint folder and csv log
    UPDATE best_name only 'best' instead full path, use cp for full
    :param config:
    :param cp_base: None for path, but usually in path/models
    :param cp_name:
    :return: path of cp, best cp, csv file path
    """
    path = config['path'] if config.get('model_path', None) is None else config['model_path']
    path = path if isinstance(path, Path) else Path(path)
    if cp_name is None:
        cp_name = "{}".format(time.strftime("%Y.%m.%d-%H.%M"))

    cp_path = path / cp_base if cp_base is not None else path
    cp_path = cp_path / cp_name
    best_name = "best"
    csvLog = cp_path / "loger"

    # write configure to easy track later, could not write JSON because some object inside config
    cp_path.mkdir(parents=True, exist_ok=True)
    txtwriter = cp_path / 'configure.txt'
    txtwriter.write_text(str(config)) if not txtwriter.is_file() else None
    try:
        with open(cp_path / 'configure.json', 'w') as f:
            json.dump(to_json_writeable(config), f, indent=2)
    except Exception as e:
        print('warning: ', e)

    return cp_path, best_name, csvLog


def map_str_prob(str_prob):
    """
    Mostly load column prob from csv or text
    :param str_prob: '[5, 3, 0, 1, 1]'
    :return: [0.5 0.3 0 .1 .1]
    """
    s = [o.replace('[', '').replace(']', '').replace(',', '').split() for o in str_prob]
    f = [[float(o) for o in row] for row in s]
    f = np.array(f)
    fn = preprocessing.normalize(f, axis=1, norm='l1')
    return fn


def map_str_list_to_type(str_lst, dtype=float):
    """
    mostly use when read list present in str format of float
    :param str_lst: [o1 o2 o3 o4 ...]
    :param dtype: type to convert, default is float
    :return:
    """
    s = str_lst.replace('[', '').replace(']', '').replace(',', '').strip().split()
    f = [dtype(o) for o in s]
    return f


def get_file_rec(path):
    """
    Get all file recursive
    :param path:
    :return: files
    """
    path = path if isinstance(path, Path) else Path(path)
    if path.is_file():
        return [path]

    ls = path.ls()
    files = [o for o in ls if o.is_file()]
    for cfolder in ls:
        if cfolder.is_dir():
            files.extend(get_file_rec(cfolder))

    return files


def load_df_img_data(path, index_keys='filename', bb_key='bb'):
    """
    Load DataFrame from csv that has filename as index and bb
    For easy to track, bb should be in format [center_x center_y width heigh]*n
    :param path:
    :param index_keys:
    :param bb_key:
    :return:
    """
    data = pd.read_csv(path, delimiter=',')
    data = data.set_index(index_keys)

    float_bb = [map_str_list_to_type(o) for o in data[bb_key]]
    data[bb_key] = float_bb

    return data


def resample_to_distribution(x, y, dis=[]):
    """
    Resampling to data x and label y to correct distribution
    :param x: data X
    :param y: label y, same size with x or None
    :param dis: list int number, number of elements get each labels, if y number then list, else dict
    :return: new_x, new_y
    """
    x = np.array(x) if isinstance(x, list) else x
    y = np.array(y) if isinstance(y, list) else y
    dis = {i: val for i, val in enumerate(dis)} if isinstance(dis, list) else dis

    assert len(x) == len(y)
    label_set = set(y)
    assert len(label_set) == (len(dis) if not isinstance(dis, dict) else len(dis.keys()))

    ss = {label: x[y == label] for label in label_set}
    x_new, y_new = np.array([]), []
    for label in label_set:
        x_new = np.append(x_new, np.random.choice(ss[label], size=dis[label]))
        y_new.extend([label] * dis[label])

    return sklearn.utils.shuffle(np.stack(x_new), np.stack(y_new))


def load_func_by_name(func_str):
    """
    Load function by full name, e.g. pcam.models.simple_transfer_model_xavier
    :param func_str: package.name
    :return: fn, module
    """
    mod_name, func_name = func_str.rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    func = getattr(mod, func_name)
    return func, mod


def save_config_info(config, save_path='.'):
    """
    Save configure (dict) to file after add more detail about date, time, git, ...
    :param config:
    :param save_path:
    :return:
    """
    current_dt = datetime.datetime.now()
    config_copy = copy.deepcopy(config)
    config_copy['save time'] = str(current_dt)

    try:
        # add git info
        from git import Repo
        repo = Repo('.')
        master = repo.heads.master
        config_copy['git_info'] = {}
        config_copy['git_info']['branch'] = repo.active_branch.name
        config_copy['git_info']['last-hash'] = repo.active_branch.log()[-1].newhexsha
        config_copy['git_info']['last-master-hash'] = master.log()[-1].newhexsha
        config_copy['git_info']['remote-url'] = repo.remotes[0].url
    except:
        pass

    with open("{}/train_config.json".format(save_path), "w") as fw:
        json.dump(config_copy, fw, indent=2, sort_keys=True)


def test_make_check_point_folder():
    print(make_check_point_folder("/tmp"))


def test_map_str_list_to_type():
    print(map_str_list_to_type("[1 3 4 6]"))
    print(map_str_list_to_type("[1 3 4 6]", dtype=int))


if __name__ == '__main__':
    test_map_str_list_to_type()

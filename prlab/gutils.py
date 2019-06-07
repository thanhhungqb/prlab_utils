import time
from pathlib import Path

import numpy as np
from sklearn import preprocessing


def make_check_point_folder(config={}, cp_base=None, cp_name=None):
    """
    make a checkpoint folder and csv log
    :param config:
    :param cp_base: None for path, but usually in path/models
    :param cp_name:
    :return: path of cp, best cp, csv file path
    """
    path = config['path']
    path = path if isinstance(path, Path) else Path(path)
    if cp_name is None:
        cp_name = "{}".format(time.strftime("%Y.%m.%d-%H.%M"))

    cp_path = path / cp_base if cp_base is not None else path
    cp_path = cp_path / cp_name
    best_name = cp_path / "best"
    csvLog = cp_path / "loger.csv"

    return cp_path, best_name, csvLog


def map_str_prob(str_prob):
    """
    Mostly load column prob from csv or text
    :param str_prob: '[5, 3, 0, 1, 1]'
    :return: [0.5 0.3 0 .1 .1]
    """
    s = [o.replace('[', '').replace(']', '').split() for o in str_prob]
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
    s = str_lst.replace('[', '').replace(']', '').strip().split()
    f = [dtype(o) for o in s]
    return f


def test_make_check_point_folder():
    print(make_check_point_folder("/tmp"))


def test_map_str_list_to_type():
    print(map_str_list_to_type("[1 3 4 6]"))
    print(map_str_list_to_type("[1 3 4 6]", dtype=int))


if __name__ == '__main__':
    test_map_str_list_to_type()

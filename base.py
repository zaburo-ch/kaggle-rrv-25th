
import tables as tb
import numpy as np
import pandas as pd
import os
import json
import datetime
from chainer import serializers
import pickle


# path

INPUT_DIR = "data/input/"
OUTPUT_DIR = "data/output/"
WORKING_DIR = "data/working/"
CON_FEATS_DIR = "data/features/continuous/"
CAT_FEATS_DIR = "data/features/categorical/"


# io

def save_array(X, filepath):
    X = np.asarray(X)
    with tb.open_file(filepath, 'w') as f:
        atom = tb.Atom.from_dtype(X.dtype)
        filters = tb.Filters(complib='blosc', complevel=9)
        ds = f.create_carray(f.root, 'X', atom, X.shape, filters=filters)
        ds[:] = X


def load_array(filepath):
    with tb.open_file(filepath, 'r') as f:
        return np.array(f.root.X)


def save_npy(X, filepath):
    X = np.asarray(X)
    np.save(filepath, X)


def load_npy(filepath):
    return np.load(filepath)


def save_df(X, filepath):
    pd.DataFrame(X).to_hdf(filepath, complevel=9, complib='blosc', key='table')


def load_df(filepath):
    return pd.read_hdf(filepath, key='table')


def save_pickle(X, filepath):
    with open(filepath, 'wb') as fp:
        pickle.dump(X, fp)


def load_pickle(filepath):
    with open(filepath, 'rb') as fp:
        ret = pickle.load(fp)
    return ret


class OutputManager:
    def __init__(self, params=None):
        self.id = None
        self.params = params
        if self.params is None:
            self.params = {}
        dt = datetime.datetime.now()
        self.params['datetime'] = dt.strftime('%Y%m%d-%H:%M:%S')
    
    def get_newest(self):
        exists = [name for name in os.listdir(OUTPUT_DIR) if len(name) == 5]
        if len(exists) == 0:
            return '00000'
        else:
            newest = sorted(map(int, exists))[-1]
            return f'{newest+1:0>5}'

    def get_path(self):
        if self.id is None:
            self.id = self.get_newest()
        dir_path = OUTPUT_DIR + self.id
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        json_path = dir_path + '/params.json'
        if self.params is not None and not os.path.exists(json_path):
            with open(json_path, 'w') as fp:
                json.dump(self.params, fp, indent=4)

        return dir_path + '/'

    def print(self, *values, filename=None):
        if filename is None:
            filename = 'log.txt'
        with open(self.get_path() + filename, 'a') as fp:
            print(*values, file=fp)
            print(*values)

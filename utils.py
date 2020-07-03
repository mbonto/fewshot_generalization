import functools
import itertools
import collections
import pickle
import torch
import numpy as np
from datasets import normalize


DEVICE = 'GPU'

def get_device():
    """Return the device to use.
    """
    if torch.cuda.is_available() and DEVICE.lower() != 'cpu':
        return torch.device("cuda:0")
    return torch.device('cpu')


class GridSearch():

    def __init__(self):
        self.ranges = dict()
        self.groups = []
        self.attributed_range = set()

    def add_range(self, name, elems):
        self.ranges[name] = elems

    def add_group(self, *args):
        self.groups.append(args)
        for arg in args:
            self.attributed_range.add(arg)

    def get_constant_keys(self):
        lst = []
        for key in self.ranges:
            if len(self.ranges[key]) == 1:
                lst.append('%s=%s'%(key,str(self.ranges[key][0])))
        return ' '.join(lst)

    def get(self, key):
        assert len(self.ranges[key]) == 1
        return self.ranges[key][0]

    def get_variable_keys(self, params):
        lst = []
        p = params._asdict()
        for key in self.ranges:
            if len(self.ranges[key]) > 1:
                lst.append('%s=%s'%(key,str(p[key])))
        return ' '.join(lst)

    def _affect_to_groups(self):
        for key in self.ranges:
            if key not in self.attributed_range:
                self.add_group(key)

    def _get_zipped(self):
        self._affect_to_groups()
        keys = []
        zippers = []
        for group in self.groups:
            keys += group
            to_zip = [self.ranges[key] for key in group]
            zipped = list(zip(*to_zip))
            zippers.append(zipped)
        return keys, zippers

    def get_params(self):
        keys, zippers = self._get_zipped()
        Params = collections.namedtuple('Params', keys)
        for unjoined in itertools.product(*zippers):
            joined = sum(unjoined, ())
            params = Params(*joined)
            yield params


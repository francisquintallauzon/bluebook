# -*- coding: utf-8 -*-
"""
Created on Thu Feb 07 13:28:14 2013

@author: francis
"""

from collections import OrderedDict
from os.path import splitext
import cPickle as pickle

class dd(OrderedDict):

    def __init__(self, *args, **kwargs):
        super(dd, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            try :
                return super(dd, self).__getattr__(key)
            except AttributeError:
                raise AttributeError, 'key = {}'.format(key)

    def __setattr__(self, key, value):

        if key in self.__dict__:
            super(dd, self).__setattr__(key, value)
            return

        if "_OrderedDict__" not in key and "_dd__" not in key:
            self[key] = value
            return

        super(dd, self).__setattr__(key, value)

    def __str__(self):
        return self.tostr(self)

    def __iadd__(self, other):
        return self.__add(self, other)

    def __add__(self, other):
        return self.__add(self.copy(), other)

    def __radd__(self, other):
        return self.__add(self.copy(), other)

    @classmethod
    def __add(cls, right, left):
        left = dd(left)
        for key, val in left.items():
            if key in right:
                right[key] += val
            else:
                right[key] = val
        return right

    def dump(self, fn, save_pretty_textfile=False):
        with open(fn, 'w') as f:
            pickle.dump(self, f)
        if save_pretty_textfile:
            self.dumptxt(splitext(fn)[0] + '.txt')
            
    def dumptxt(self, fn):
        with open(fn, 'w') as f:
            f.write(str(self))

    @classmethod
    def load(cls, fn):
        with open(fn, 'r') as f:
            return pickle.load(f)

    @classmethod
    def tostr(cls, d=None, prefix=''):
        s = ""
        for key, value in d.items():
            if isinstance(value, dict):
                s += cls.tostr(value, prefix + str(key) + '.')
            else:
                s += prefix + str(key) + '=' + str(value) + '\n'
        return s


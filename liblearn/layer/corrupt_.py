# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 16:16:15 2015

@author: francis
"""

import theano                       as th
from datetime                       import datetime
from liblearn                 import base
from liblearn.utils           import cast_x, float_x


class corrupt(base):

    rng = th.tensor.shared_randomstreams.RandomStreams(datetime.now().microsecond)

    def __init__(self, corruption_type, corruption_level):
        super(corrupt, self).__init__()
        self.add_hparam(corruption_type=corruption_type)
        self.add_hparam(corruption_level=corruption_level)

    def __call__(self, inp, mode=None):

        corruption_type=self.corruption_type
        corruption_level=self.corruption_level

        if mode != 'train':
            print('corrupt : mode (= {}) != "train"'.format(mode))
            return inp
        elif corruption_level==0 or corruption_type == None:
            return inp
        elif corruption_type=='zeromask':
            return self.rng.binomial(size=inp.shape, n=1, p=1.0-corruption_level, dtype=float_x) * inp / cast_x(1-corruption_level)
        elif corruption_type=='gaussian':
            return self.rng.normal(size=inp.shape, avg=0.0, std=corruption_level, dtype=float_x) + inp
        else :
            raise ValueError


    def __str__(self):
        return '{} using {} level {} noise'.format(self.name, self.corruption_level, self.corruption_type)
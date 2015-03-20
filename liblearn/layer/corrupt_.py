# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 16:16:15 2015

@author: francis
"""

import theano                       as th
from datetime                       import datetime
from liblearn                 import base
from liblearn.utils           import cast_x


class corrupt(base):

    rng = th.tensor.shared_randomstreams.RandomStreams(datetime.now().microsecond)
    
    def __init__(self, corruption_type, corruption_level):
        super(corrupt, self).__init__({'corruption_type':corruption_type, 'corruption_level':corruption_level})

    def __call__(self, inp):
        corruption_type=self.corruption_type
        corruption_level=self.corruption_level
        if corruption_level==0 or corruption_level==None or corruption_type==None or corruption_type=='none' or corruption_type=='None':
            return inp
        elif corruption_type=='zeromask':
            return self.rng.binomial(size=inp.shape, n=1, p=1.0-corruption_level, dtype=th.config.floatX) * inp / cast_x(1-corruption_level)
        elif corruption_type=='gaussian':
            return self.rng.normal(size=inp.shape, avg=0.0, std=corruption_level, dtype=th.config.floatX) + inp

        
    def __str__(self):
        return '{} using {} level {} noise'.format(self.name, self.corruption_level, self.corruption_type)
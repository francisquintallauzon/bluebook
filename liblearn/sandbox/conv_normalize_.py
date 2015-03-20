# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 15:36:18 2014

@author: francis
"""

import theano.tensor          as T
from liblearn.utils     import cast_x
from liblearn.layer     import base

def conv_normalize(inp):
    return inp / T.sqrt((inp**cast_x(2)).sum(3).sum(2).sum(1)).dimshuffle(0,'x','x','x') + cast_x(0.00001)

# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 10:40:27 2014

@author: francis
"""

import theano.tensor as T
from learn.utils                    import cast_x

def normalize(inp):
    return inp / T.sqrt((inp**cast_x(2)).sum(3).sum(2).sum(1)).dimshuffle(0,'x','x','x') + cast_x(0.00001)

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 13:22:05 2014

@author: francis
"""

import numpy as np
import theano as th
import theano.tensor as T

float_x = th.config.floatX

def cast_x(inp):
    return T.cast(inp, th.config.floatX)

def shared_x(value, name=None):
    return value if isinstance(value, T.sharedvar.SharedVariable) else th.shared(np.asarray(value, float_x), name=name)

def shared_zeros_like(shared, name=None, strict=False, allow_downcast=None):
    return th.shared(np.zeros_like(shared.get_value()), name, strict, allow_downcast)

def shared_copy(copy_from, copy_to=False):
    if copy_to :
        copy_to.set_value(copy_from.get_value())
    else :
        copy_to = th.shared(copy_from.get_value())
    return copy_to


# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 22:52:17 2013

@author: francis
"""

import numpy          as np
import theano.tensor  as T

from numpy.random     import uniform
from math             import sqrt
from liblearn         import base
from liblearn.utils   import shared_x
from liblearn.utils   import activation

class hidden(base):

    def __init__(self, act, nb_inp, nb_hid, W=None, b=None):
        
        super(hidden, self).__init__()

        self.add_hparam(act=act)
        self.add_hparam(nb_inp=nb_inp)
        self.add_hparam(nb_hid=nb_hid)
        
        self.add_param(W = shared_x(uniform(-1/sqrt(self.nb_inp), 1/sqrt(self.nb_inp), (self.nb_inp, nb_hid)), name='W') if W is None else W, optimizable=True)
        self.add_param(b = shared_x(np.zeros(nb_hid), name='b') if b is None else b, optimizable=True)
        

    def __call__(self, inp, mode):
        act = activation(self.act)
        return act(T.dot(inp.flatten(2), self.W) + self.b)

        
    def shape(self, input_sz=None):
        return (input_sz[0], self.nb_hid)
            
    def __str__(self):
        return '{} with {} filters and {} activation'.format(self.name, (self.nb_inp, self.nb_hid), self.act)


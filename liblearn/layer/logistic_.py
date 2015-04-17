# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 22:55:06 2013

@author: francis
"""

import numpy                        as np
import theano.tensor                as T
from liblearn                 import base
from liblearn.utils           import shared_x, float_x

class logistic(base):

    def __init__(self, nb_inp, nb_out, W=None, b=None):
        """ Logistic regression layer """

        super(logistic, self).__init__()    

        self.add_hparam(nb_inp=nb_inp)
        self.add_hparam(nb_out=nb_out)
        
        self.add_param(W = shared_x(np.zeros((self.nb_inp, self.nb_out), float_x), name='W') if W is None else W, optimizable=True)
        self.add_param(b = shared_x(np.zeros(self.nb_out, float_x), name='b') if b is None else b, optimizable=True)
        

    def __call__(self, inp, mode=None):
        return T.nnet.softmax(T.dot(inp, self.W) + self.b)
        
    def __str__(self):
        return '{} with {} filters'.format(self.name, ((self.nb_inp, self.nb_out)))
        

    def shape(self, input_sz):
        return (input_sz[0], self.nb_out)

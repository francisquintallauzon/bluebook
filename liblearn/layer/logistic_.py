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
        

        hparams = {'nb_inp':nb_inp,
                   'nb_out':nb_out} 
        
        params = {'b':b,
                  'W':W}        
        
        super(logistic, self).__init__(hparams, params)        
        
        # Optimizable parameters
        self.b = b if b else shared_x(np.zeros(self.nb_out, float_x), name='b')
        self.W = W if W else shared_x(np.zeros((self.nb_inp, self.nb_out), float_x), name='W')

    def __call__(self, inp, dropout_level=0):
        return T.nnet.softmax(T.dot(inp, self.W) + self.b)
        
    def __str__(self):
        return '{} with {} filters'.format(self.name, ((self.nb_inp, self.nb_out)))
        

    def shape(self, input_sz):
        return (input_sz[0], self.nb_out)

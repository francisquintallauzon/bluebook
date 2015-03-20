# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 22:52:17 2013

@author: francis
"""

import numpy                as np
import theano.tensor        as T

from math                   import sqrt
from liblearn         import base
from liblearn.utils   import shared_x
from liblearn.utils   import activation

class hidden(base):

    def __init__(self, act, nb_inp, nb_hid, init='uniform', normalize=False, W=None, b=None):

        hparams = {'act':act,
                   'nb_inp':nb_inp,
                   'nb_hid':nb_hid,
                   'init':init,
                   'normalize':normalize} 
        
        params = {'b':b,
                  'W':W}        
        
        super(hidden, self).__init__(hparams, params)

        if W is None:
            if init=='gaussian':
                W = np.random.normal(0, 0.01, (self.nb_inp, nb_hid))
            elif init=='uniform':
                W = np.random.uniform(-1/sqrt(self.nb_inp), 1/sqrt(self.nb_inp), (self.nb_inp, nb_hid))

            if normalize:
                W /= np.sqrt((W**2).sum(0))[None, :]

            self.W = shared_x(W, name='W')

        if b is None:
            self.b = shared_x(np.zeros(nb_hid), name='b')


    def __call__(self, inp):
        act = activation(self.act)
        return act(T.dot(inp.flatten(2), self.W) + self.b)

        
    def shape(self, input_sz=None):
        return (input_sz[0], self.nb_hid)
            

        
    def __str__(self):
        return '{} with {} filters and {} activation'.format(self.name, (self.nb_inp, self.nb_hid), self.act)


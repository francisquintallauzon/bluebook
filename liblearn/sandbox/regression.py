# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 15:19:55 2013

@author: francis
"""

import numpy                        as np
import theano                       as th
import theano.tensor                as T
from learn.utils.corrupt            import corrupt

class regression(object):

    def __init__(self, inp, labels, nb_in, nb_out, dropout_level=0):
        """
        numeric regression layer
        """

        ###########################################################################################
        # Storage
        self.nb_in = nb_in
        self.nb_out = nb_out
        self.dropout_level = dropout_level

        ###########################################################################################
        # Learn model
        
        # Weight initialization
        self.B = th.shared(np.zeros(nb_out, dtype = th.config.floatX), borrow=False, name='Numeric regression biases')
        self.W = th.shared(np.random.uniform(low=-1./np.sqrt(nb_in), high=1./np.sqrt(nb_in), size=(nb_in, nb_out)).astype(th.config.floatX), borrow=False, name='Numeric regression weights')

        # Corrupted model
        noisy_inp = corrupt(inp, 'zeromask' if dropout_level > 0 else None, dropout_level)
        self.noisy_pred = T.dot(noisy_inp, self.W) + self.B
        self.noisy_cost = T.sum((0.5*(self.noisy_pred - labels)**2).sum(axis = 1))
        
        # Clean model
        clean_inp = inp
        self.clean_pred = (T.dot(clean_inp, self.W) + self.B) / T.cast(1-dropout_level, th.config.floatX)
        self.clean_cost = T.mean((0.5*(self.clean_pred - labels)**2).sum(axis = 1))
        
        # Prediction model
        self.pred = self.clean_pred

        # Prediction error (for compatibility with supervised learning algorithm)
        self.error = T.mean(abs(self.clean_pred - labels))

        ###########################################################################################
        # For for interactions with other models
        self.inp = inp
        self.labels = labels
        self.out = self.clean_pred
        self.params = [self.W, self.B]
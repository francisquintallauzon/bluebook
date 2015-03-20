# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 15:43:28 2015

@author: francis
"""

import theano.tensor as T
from liblearn import base

class Error(base):
    def __call__(self, prob, labels):
        return T.mean(T.neq(T.argmax(prob, axis=1), labels.flatten()))
    
error = Error()

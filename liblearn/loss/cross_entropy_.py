# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 14:40:02 2015

@author: francis
"""

import theano.tensor as T
from liblearn  import base

class Cross_entropy(base):
    def __call__(self, prob, labels):
        return -T.mean(T.log(prob[T.arange(labels.shape[0], dtype='int32'), labels.flatten()]))

cross_entropy = Cross_entropy()


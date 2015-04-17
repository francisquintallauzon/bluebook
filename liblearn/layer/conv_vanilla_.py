# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 22:52:17 2013

@author: francis
"""

import numpy                        as np
from theano.tensor.nnet.conv        import conv2d as conv
from numpy.random                   import uniform
from liblearn                       import base
from liblearn.utils                 import activation, shared_x

class conv_vanilla(base):
    
    def __init__(self, act, filter_sz, W = None, b = None):
        """
        Vanilla convolutional layer.  
        
        Parameters
        ----------
                            
        activation          theano functor or function that return a theano object
                            layer's activation function 

        filter_sz           4-dimension tuple of integers
                            filter size (nb filters, stack size, nb lines, nb columns)

        W                   th.shared or None
                            convolutional layer weights

        b                   th.shared
                            convolutional layer biases
        """

        super(conv_vanilla, self).__init__()
        
        self.add_hparam(act=act)
        self.add_hparam(filter_sz=filter_sz)
        self.add_param(b=shared_x(np.zeros(filter_sz[0]), name='b') if b is None else b, optimizable=True)
        self.add_param(W=shared_x(uniform(-0.1, 0.1, size=filter_sz), name='W') if W is None else W, optimizable=True)
        
        
    def __call__(self, inp, mode=None):
        act = activation(self.act)
        return act(conv(inp, self.W) + self.b.dimshuffle('x', 0, 'x', 'x'))


    def __str__(self):
        return '{} with {} filters and {} activation'.format(self.name, self.filter_sz, self.act)
        
        
    def shape(self, input_sz):
        return (input_sz[0], self.filter_sz[0], input_sz[2]-self.filter_sz[2]+1, input_sz[2]-self.filter_sz[2]+1)

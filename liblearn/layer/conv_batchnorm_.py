# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 17:25:09 2015
@author: francis
"""

import numpy                        as np
import theano                       as th
import theano.tensor                as T
from liblearn                       import base
from liblearn.utils                 import cast_x, shared_x

class conv_batchnorm(base):

    def __init__(self, filter_sz, beta=None, gamma=None, mean=None, var=None):
        super(conv_batchnorm, self).__init__()
        self.add_hparam(filter_sz=filter_sz)
        self.add_param(beta = shared_x(np.zeros(filter_sz), name='beta') if beta is None else beta, optimizable=True)
        self.add_param(gamma = shared_x(np.ones(filter_sz), name='gamma') if gamma is None else gamma, optimizable=True)
        self.add_param(mean = shared_x(np.zeros(filter_sz), name='mean') if mean is None else mean, optimizable=False)
        self.add_param(var = shared_x(np.zeros(filter_sz), name='var') if var is None else var, optimizable=False)


    def learn(self, model_inp, layer_inp, data):

        print('    Learning {}'.format(self.name))

        count = shared_x(0., name='count')
        updates  = [(count, count + cast_x(layer_inp.shape[0]))]
        updates += [(self.mean, self.mean + layer_inp.sum(0))]
        updates += [(self.var, self.var + (layer_inp**2).sum(0))]
        fn = th.function(inputs=[model_inp], updates = updates)
        for i, (example, label) in enumerate(data):
            fn(example)
        self.mean.set_value((self.mean/count).eval())
        self.var.set_value((self.var/count - self.mean**2).eval())

        print('      - mean:  mean(mean) = {:0.2f};  std(mean) = {:0.2f}'.format(float(self.mean.mean().eval()), float(self.mean.std().eval())))
        print('      - var:  mean(var) = {:0.2f};  std(var) = {:0.2f}'.format(float(self.var.mean().eval()), float(self.var.std().eval())))


    def __call__(self, inp, mode):
        if mode == 'train' or mode == 'valid':
            inp = inp-inp.mean(0).dimshuffle('x', 0, 1, 2)
            inp = inp / T.sqrt((inp**cast_x(2)).mean(0).dimshuffle('x', 0, 1, 2) + cast_x(0.0001))
            return inp * self.gamma + self.beta
        else :
            mean = self.mean.dimshuffle('x', 0, 1, 2)
            std = T.sqrt(self.var + cast_x(0.0001)).dimshuffle('x', 0, 1, 2)
            beta = self.beta.dimshuffle('x', 0, 1, 2)
            gamma = self.gamma.dimshuffle('x', 0, 1, 2)
            return inp*gamma/std  + beta - mean*gamma/std


if __name__ == '__main__':
    pass
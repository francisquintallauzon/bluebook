# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 10:56:20 2014

@author: francis
"""

if __name__ == '__main__':
    import sys
    sys.path += '../../'

import numpy                        as np
import theano                       as th
import theano.tensor                as T
from liblearn                       import base
from liblearn.utils                 import cast_x, shared_x

class conv_preprocess(base):

    def __init__(self, nb_channels, nb_pretrain_iterations = 1e99, dc=None, std=None):
        super(conv_preprocess, self).__init__()

        self.add_hparam(nb_channels=nb_channels)
        self.add_hparam(nb_pretrain_iterations=nb_pretrain_iterations)

        self.add_param(dc = shared_x(np.zeros(nb_channels), name='dc') if dc is None else dc, optimizable=False)
        self.add_param(std = shared_x(np.zeros(nb_channels), name='std') if std is None else std, optimizable=False)

    def learn(self, model_inp, layer_inp, data):

        print('    Learning {}'.format(self.__class__.__name__))

        count = shared_x(0., name='count')
        updates  = [(count, count + cast_x(layer_inp.shape[0]))]
        updates += [(self.dc, self.dc + layer_inp.mean(3).mean(2).sum(0))]
        updates += [(self.std, self.std + (layer_inp**2).mean(3).mean(2).sum(0))]
        fn = th.function(inputs=[model_inp], updates = updates)
        for i, (example, label) in enumerate(data):
            if i >= self.nb_pretrain_iterations:
                break
            fn(example)
        self.dc.set_value((self.dc/count).eval())
        self.std.set_value(T.sqrt(self.std/count - self.dc**2).eval())

        print('      - dc centering:  mean(dc) = {:0.2f};  std(dc) = {:0.2f}'.format(float(self.dc.mean().eval()), float(self.dc.std().eval())))
        print('      - contrast nrm:  mean(std) = {:0.2f};  std(std) = {:0.2f}'.format(float(self.std.mean().eval()), float(self.std.std().eval())))

    def __call__(self, inp, mode=None):
        return (inp-self.dc.dimshuffle('x', 0, 'x', 'x')) / self.std.dimshuffle('x', 0, 'x', 'x')



if __name__ == '__main__':
    test = conv_preprocess(1, 1)
    test.export("./conv_preprocess_test")
    test.load("./conv_preprocess_test")

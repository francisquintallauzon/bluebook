# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 10:05:08 2014

@author: francis
"""


from theano.tensor.signal.downsample import max_pool_2d
from liblearn                        import base

class conv_maxpool(base):
    def __init__(self, downsample_sz):
        super(conv_maxpool, self).__init__()
        self.add_hparam(downsample_sz=(downsample_sz, downsample_sz) if isinstance(downsample_sz, int) else downsample_sz)

    def __call__(self, inp, mode=None):
        return max_pool_2d(inp, self.downsample_sz, ignore_border=True)

    def shape(self, input_sz):
        return (input_sz[0], input_sz[1], int(input_sz[2]/self.downsample_sz[0]), int(input_sz[3]/self.downsample_sz[1]))

    def __str__(self):
        return '{} with {} downsampling'.format(self.name, self.downsample_sz)
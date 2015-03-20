# -*- coding: utf-8 -*-
"""
Implements a feedback autoencoder

@author: francis
"""

if __name__ == '__main__':
    import sys
    sys.path.append("../../")

import theano                       as th
import numpy                        as np
import theano.tensor                as T
from os.path                        import join
from numpy.random                   import uniform
from learn.utils                    import corrupt
from utils.path                     import make_dir
from learn.utils                    import filterstoimg
from learn.utils                    import step
from learn.layer                    import base
from learn.utils                    import shared_x

def feedback(base):
    def __init__(self, threshold, nb_inp, nb_hid, W=None, layer_id=''):
        base.__init__(self, layer_id, W=W)

        # Hyperparameters
        self.threshold = threshold
        self.nb_inp = nb_inp
        self.nb_hid = nb_hid

        # Set optimizable parameter dictionary
        if self.W == None:
            init = uniform(low=-1, high=1, size=(nb_inp, nb_hid)).astype(th.config.floatX)
            init = init/np.sqrt((init**2).sum(0))[None,:]
            self.W = shared_x(init, name='W'+self.layer_id)



    def __call__(self,):
        pass





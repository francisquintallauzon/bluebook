# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 22:52:17 2013

@author: francis
"""

import os
import theano                       as th
import numpy                        as np
import theano.tensor                as T

from theano.tensor.nnet.conv        import conv2d as conv
from numpy.random                   import uniform
from learn.utils.corrupt            import corrupt
from utils.path                     import make_dir
from learn.utils.display            import convfilterstoimg
from learn.utils.display            import filterstoimg

class conv_vanilla(object):
    def __init__(self, params, act, model_inp, layer_inp, corruption_type=None, corruption_level=0):

        # Parameters
        self.params = params
        self.model_inp = model_inp
        self.layer_inp = layer_inp
        self.corruption_type = corruption_type
        self.corruption_level = corruption_level

        # Model
        corr_inp = corrupt(layer_inp, corruption_type, corruption_level)
        self.out = act(conv(corr_inp, params.W, border_mode='valid') + params.B.dimshuffle('x', 0, 'x', 'x'))


    def output_sz(self, input_sz):
        # Compute output size
        filter_sz = self.params.filter_sz
        return (input_sz[0], filter_sz[0], input_sz[2]-filter_sz[2]+1, input_sz[3]-filter_sz[3]+1)


class conv_vanilla_params(object):
    def __init__(self, act, filter_sz, layer_id=None):
        """
        Initializes convolutional filters.

        Parameters
        ----------
        act :           activation type
                        activation object
        filter_sz:      4-dimension tuple of integers
                        filter size (nb filters, stack size, nb lines, nb columns)
                        for input layer : stack size == nb image channels
        layer_id        string representable type or NoneType
                        layer identification (useful for debugging)
        """

        # Make sure all size dimension are positive, otherwise raise an error
        if np.any(np.asarray(filter_sz) <= 0):
            raise ValueError, "Negative value in filter_sz (={})".format(filter_sz)

        # Hyperparameters
        self.act = act
        self.filter_sz = filter_sz

        # For debug
        self.__debug_enabled = False
        self.__pca_weights = None

        # Format layer_id
        self.layer_id = '' if layer_id == None else '_' + str(layer_id)

        # Parameters
        factor = 4. if act == T.nnet.sigmoid else 1.
        nb_inp = filter_sz[1] * filter_sz[2] * filter_sz[3]
        bound = factor*np.sqrt(6./nb_inp)
        self.W = th.shared(uniform(low=-bound, high=bound, size=filter_sz).astype(th.config.floatX), borrow=True, name='conv_vanilla_filters{}'.format(self.layer_id))
        self.B = th.shared(np.zeros(filter_sz[0], dtype = th.config.floatX), borrow=True, name='conv_vanilla_biases{}'.format(self.layer_id))

        # Internal
        self._params = [self.W, self.B]

    def __call__(self):
        """
        Returns a list of selected parameters
        """
        return self._params

    def export(self, path):
        """
        Export selected parameters matrices to npy files
        """
        make_dir(path)
        for param in self._params:
            fn = os.path.join(path, "{}.npy".format(str(param)))
            np.save(fn, param.get_value(borrow=True))

    def set_debug_info(self, debug_path, pca_weights=None, prefix=None):
        make_dir(debug_path)
        self.__debug_enabled = True
        self.__pca_weights = pca_weights
        self.__debug_path = debug_path
        self.__prefix = '' if prefix == None else prefix  + '_'


    def debug_call(self):

        if self.__debug_enabled:

            if self.__pca_weights == None:
                convfilterstoimg(self.W.get_value(), fn = os.path.join(self.__debug_path, self.__prefix + str(self.W) + '.tif'))
            else:
                w = self.W.get_value()
                pca = self.__pca_weights
                patch_sz = np.asarray(pca.shape[1:])[[1,2,0]]

                if tuple(w.shape[2:]) != (1,1):
                    raise ValueError, "w.shape[2:] is not (1,1).  Instead w.shape = {}".format(w.shape)

                if w.shape[1] != pca.shape[0]:
                    raise ValueError, "w.shape[1] is different from pca.shape[0]. Instead, w.shape = {} and pca.shape = {}".format(w.shape, pca.shape)

                pca = self.__pca_weights.reshape((self.__pca_weights.shape[0], -1)).T
                w = w.squeeze((2,3)).T

                # print "pca.shape = ", pca.shape, "w.shape = ", w.shape, 'patch_sz = ', patch_sz
                filterstoimg(np.dot(pca, w), patch_sz, fn = os.path.join(self.__debug_path, self.__prefix + str(self.W) + '.tif'))


    def fromfile(self, path):

        if path == None:
            return

        for param in self._params:
            fn = os.path.join(path, "{}.npy".format(param))
            val = np.load(fn)
            print 'Loading parameter {} with shape {}'.format(param, val.shape)
            param.set_value(val)

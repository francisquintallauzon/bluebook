# -*- coding: utf-8 -*-
"""
mean-covariance convolutional layer

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


class conv_mc(object):
    def __init__(self, params, model_inp, layer_inp, corruption_type=None, corruption_level=0):

        # Parameters
        self.params = params
        self.model_inp = model_inp
        self.layer_inp = layer_inp
        self.corruption_type = corruption_type
        self.corruption_level = corruption_level

        # corrupt input
        corr_inp = corrupt(layer_inp, corruption_type, corruption_level)

        out = []
        self.nb_channels = 0

        if params.mean_filter_size!= 0:
            out += [params.m_act(conv(corr_inp, params.mean_filters) + params.mean_b.dimshuffle('x', 0, 'x', 'x'))]
            self.nb_channels += params.mean_filter_size[0]

        if params.cov_filter_size != 0:
            f = conv(corr_inp, params.cov_filters)**2
            out += [params.c_act(conv(f, params.cov_mapping) + params.cov_b.dimshuffle('x', 0, 'x', 'x'))]
            self.nb_channels += params.map_filter_size[0]

        self.out = T.concatenate(out, axis=1)

    def output_sz(self, input_sz):
        return (input_sz[0], self.nb_channels, input_sz[2]-self.params.mean_filter_size[2]+1, input_sz[3]-self.params.mean_filter_size[3]+1)


class conv_mc_params(object):
    def __init__(self, c_act, m_act, cov_filter_size, mean_filter_size, nb_map, layer_id=None):
        """
        Initializes gated dae parameters

        Parameters
        ----------
        c_act, m_act:       activation function type
                            AE activation for covariance : c_act; mean : m_act
        cov_filter_size:    4-dimension tuple of integers
                            cov filter size (nb filters, stack size, nb lines, nb columns)
                            for input layer : stack size == nb image channels
        mean_filter_size:   4-dimension tuple of integers
                            mean filter size (nb filters, stack size, nb lines, nb columns)
                            for input layer : stack size == nb image channels
        nb_map              integer type
                            number of mapping units
        layer_id            string representable type or NoneType
                            layer identification (useful for debugging)
        """

        # Make sure all size dimension are positive, otherwise raise an error
        if np.any(np.asarray(cov_filter_size) <= 0):
            raise ValueError, "Negative value in cov_filter_size (={})".format(cov_filter_size)
        if np.any(np.asarray(mean_filter_size) <= 0):
            raise ValueError, "Negative value in mean_filter_size (={})".format(mean_filter_size)

        # For debug
        self.__debug_enabled = False
        self.__pca_weights = None

        # Format layer_id
        self.layer_id = '' if layer_id == None else '_' + str(layer_id)

        # Hyperparameters
        self.c_act = c_act
        self.m_act = m_act
        self.cov_filter_size = cov_filter_size
        self.mean_filter_size = mean_filter_size
        self.map_filter_size = (nb_map, cov_filter_size[0], 1, 1)

        # Internal list of parameters
        self._params = []

        # Format layer_id
        layer_id = '' if layer_id==None else '_' + str(layer_id)

        if mean_filter_size != None:
            bound = (4. if m_act == T.nnet.sigmoid else 1.) * np.sqrt(6./(mean_filter_size[1] * mean_filter_size[2] * mean_filter_size[3]))
            self.mean_filters = th.shared(uniform(low =-bound, high=bound, size=mean_filter_size).astype(th.config.floatX), name='mean_filters{}'.format(layer_id))
            self.mean_b = th.shared(np.zeros(mean_filter_size[0], dtype = th.config.floatX), name='mean_biases{}'.format(layer_id))
            self._params += [self.mean_filters, self.mean_b]

        if cov_filter_size != None:
            bound = (4. if c_act == T.nnet.sigmoid else 1.) * np.sqrt(6./(cov_filter_size[1] * cov_filter_size[2] * cov_filter_size[3]))
            self.cov_filters = th.shared(uniform(low=-bound, high=bound, size=cov_filter_size).astype(th.config.floatX), name='cov_filters{}'.format(layer_id))

            bound = (4. if m_act == T.nnet.sigmoid else 1.) * np.sqrt(6./cov_filter_size[0])
            self.cov_mapping = th.shared(uniform(low=-bound, high=bound, size=(nb_map, cov_filter_size[0], 1, 1)).astype(th.config.floatX), name='mapping_filters{}'.format(layer_id))

            self.cov_b = th.shared(np.zeros(nb_map, dtype = th.config.floatX), name='mapping_biases{}'.format(layer_id))

            self._params += [self.cov_filters, self.cov_mapping, self.cov_b]


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
                if self.mean_filter_size != None:
                    convfilterstoimg(self.mean_filters.get_value(), fn = os.path.join(self.__debug_path, self.__prefix + str(self.mean_filters) + '.tif'))
                if self.cov_filter_size != None:
                    convfilterstoimg(self.cov_filters.get_value(), fn = os.path.join(self.__debug_path, self.__prefix + str(self.cov_filters) + '.tif'))
            else:
                pca = self.__pca_weights
                patch_sz = np.asarray(pca.shape[1:])[[1,2,0]]
                pca = self.__pca_weights.reshape((self.__pca_weights.shape[0], -1)).T

                # Process mean_filters
                mean_filters = self.mean_filters.get_value()
                if tuple(mean_filters.shape[2:]) != (1,1):
                    raise ValueError, "mean_filters.shape[2:] is not (1,1).  Instead mean_filters.shape = {}".format(mean_filters.shape)
                if mean_filters.shape[1] != pca.shape[1]:
                    raise ValueError, "mean_filters.shape[1] is different from pca.shape[1]. Instead, mean_filters.shape = {} and pca.shape = {}".format(mean_filters.shape, pca.shape)
                mean_filters = mean_filters.squeeze((2,3)).T
                # print "pca.shape = ", pca.shape, "mean_filters.shape = ", mean_filters.shape, 'patch_sz = ', patch_sz
                filterstoimg(np.dot(pca, mean_filters), patch_sz, fn = os.path.join(self.__debug_path, self.__prefix + str(self.mean_filters) + '.tif'))

                # Process cov_filters
                cov_filters = self.cov_filters.get_value()
                if tuple(cov_filters.shape[2:]) != (1,1):
                    raise ValueError, "cov_filters.shape[2:] is not (1,1).  Instead cov_filters.shape = {}".format(cov_filters.shape)
                if cov_filters.shape[1] != pca.shape[1]:
                    raise ValueError, "cov_filters.shape[1] is different from pca.shape[1]. Instead, cov_filters.shape = {} and pca.shape = {}".format(cov_filters.shape, pca.shape)
                cov_filters = cov_filters.squeeze((2,3)).T
                # print "pca.shape = ", pca.shape, "cov_filters.shape = ", cov_filters.shape, 'patch_sz = ', patch_sz
                filterstoimg(np.dot(pca, cov_filters), patch_sz, fn = os.path.join(self.__debug_path, self.__prefix + str(self.cov_filters) + '.tif'))


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
from learn.train.sgd                import sgd
from learn.utils.corrupt            import corrupt
from utils.path                     import make_dir
from learn.utils.display            import convfilterstoimg

class conv_dae(object):
    def __init__(self, params, act, model_inp, layer_inp, input_sz, filter_sz, inp_corruption_type=None, inp_corruption_level=0, hid_corruption_type=None, hid_corruption_level=0, L1_hiddens = 0, L2_weights = 0):

        # Parameters
        self.act = act
        self.params = params
        self.model_inp = model_inp
        self.layer_inp = layer_inp
        self.input_sz = input_sz
        self.filter_sz = filter_sz
        self.inp_corruption_type = inp_corruption_type
        self.inp_corruption_level = inp_corruption_level
        self.hid_corruption_type = hid_corruption_type
        self.hid_corruption_level = hid_corruption_level
        self.L1_hiddens = L1_hiddens
        self.L2_weights = L2_weights
        self.debug = False


    def _model(self, inp, inp_corruption_type=None, inp_corruption_level=0, hid_corruption_type=None, hid_corruption_level=0, L1_hiddens = 0, L2_weights = 0):

        # Encoder
        #enc_input_sz = self.input_sz
        #enc_filter_sz = self.filter_sz
        corr_inp = corrupt(inp, inp_corruption_type, inp_corruption_level)
        hid = self.act(conv(corr_inp, self.params.w_enc, border_mode='valid') + self.params.b_enc.dimshuffle('x', 0, 'x', 'x'))

        # Decoder
        #dec_input_sz = (enc_input_sz[0], enc_filter_sz[0], enc_input_sz[2]-enc_filter_sz[2]+1, enc_input_sz[3]-enc_filter_sz[3]+1)
        #dec_filter_sz = (int(np.prod(enc_input_sz[1:])), enc_filter_sz[0], 1, 1)
        corr_hid = corrupt(hid, hid_corruption_type, hid_corruption_level)
        out = conv(corr_hid, self.params.w_dec, border_mode='valid')

        # Make cost function
        cost = T.mean((0.5*(out.flatten(2) - inp.flatten(2))**2).sum(axis = 1))

        # Add L1 hiddens cost
        if L1_hiddens > 0:
            cost += L1_hiddens * abs(hid).sum(1).mean()

        # Add L2 weight cost
        if L2_weights > 0:
            cost += L2_weights * ((self.params.w_enc**2.).sum() +
                                  (self.params.w_dec**2.).sum())

        return hid, cost


    @property
    def out(self):
        # Returns encoder output
        hid, _ = self._model(self.layer_inp, self.inp_corruption_type, self.inp_corruption_level, self.hid_corruption_type, self.hid_corruption_level, self.L1_hiddens, self.L2_weights)
        return hid


    @property
    def output_sz(self):
        # Compute output size
        return (self.input_sz[0], self.filter_sz[0], self.input_sz[2]-self.filter_sz[2]+1, self.input_sz[3]-self.filter_sz[3]+1)


    def learn(self, lookback, max_epoch, minibatch_sz, train_batch_size_ratio, valid_batch_size_ratio, learning_rate, momentum, fetcher, fetchonce=True, maxnorm=0, do_validation=True, decay_rate=1.0):

        if tuple(self.output_sz[2:]) != (1,1) :
            raise ValueError, "At learn time, self.output_sz[2,3] should be (1,1).  Instead it is {}".format(self.output_sz)

        # Non-denoising version of the autoencoder (for measuring validation cost)
        inp_scale = 1-self.inp_corruption_level if self.inp_corruption_type == 'zeromask' else 1.
        _, clean_cost = self._model(self.layer_inp*inp_scale)

        # Noisy version of the gated ae
        _, noisy_cost = self._model(self.layer_inp, self.inp_corruption_type, self.inp_corruption_level, self.hid_corruption_type, self.hid_corruption_level, self.L1_hiddens, self.L2_weights)

        # Initialize learning
        self.trainer = sgd(self.model_inp, None, noisy_cost, None, clean_cost, None, self.params(), maxnorm, self._debug_call)

        # Learn
        self.trainer.learn(lookback, max_epoch, minibatch_sz, train_batch_size_ratio, valid_batch_size_ratio, learning_rate, momentum, fetcher, fetchonce, do_validation, decay_rate)


    def _set_debug_info(self, pca_weights, debug_path):
        make_dir(debug_path)
        self.debug = True
        self.pca_weights = pca_weights
        self.debug_path = debug_path


    def _debug_call(self):
        if self.debug:
            pca_shape = self.pca_weights.shape
            w_enc = self.params.w_enc.get_value()
            flat_w = w_enc.reshape((w_enc.shape[0], -1))
            flat_pca = self.pca_weights.reshape((pca_shape[0], -1))
            filters = np.dot(flat_w, flat_pca)
            filters = filters.reshape((w_enc.shape[0], pca_shape[1], pca_shape[2], pca_shape[3]))
            convfilterstoimg(filters, fn = os.path.join(self.debug_path, 'conv_dae_layer_{}.tif'.format(self.params.layer_id)))

    def clear(self):
        del self.trainer


class conv_dae_params(object):
    def __init__(self, act, filter_sz, layer_id='', tied_weights = True):
        """
        Initializes gated dae parameters

        Parameters
        ----------
        act :           activation type
                        activation object
        filter_sz:      4-dimension tuple of integers
                        filter size (nb filters, stack size, nb lines, nb columns)
                        for input layer : stack size == nb image channels
        layer_id        string representable type
                        string identification (useful for debugging)
        tied_weights    logical
                        if true, encoder and decoder weights are tied.  They are
                        not tied otherwise

        """

        # Hyperparameters
        self.act = act
        self.filter_sz = filter_sz
        self.layer_id = layer_id

        # List of parameters
        self._params = []

        # encoder bias
        self.b_enc = th.shared(np.zeros(filter_sz[0], dtype = th.config.floatX), borrow=True, name='encoder_biases_'+str(layer_id))
        self._params += [self.b_enc]

        # encoder weights
        nb_inp = filter_sz[1] * filter_sz[2] * filter_sz[3]
        factor = 4. if act == T.nnet.sigmoid else 1.
        bound = factor * np.sqrt(6. / nb_inp)
        self.w_enc = th.shared(uniform(low=-bound, high=bound, size=filter_sz).astype(th.config.floatX), borrow=True, name='encoder_weights_' + str(layer_id))
        self._params += [self.w_enc]

        # decoder weights
        if tied_weights:
            self.w_dec = self.w_enc.dimshuffle(0, 2, 3, 1).flatten(2).dimshuffle(1, 0, 'x', 'x')
        else:
            filter_sz = [nb_inp, filter_sz[0], 1, 1]
            factor = 4. if act == T.nnet.sigmoid else 1.
            bound = factor * np.sqrt(6. / filter_sz[0])
            self.w_dec = th.shared(uniform(low=-bound, high=bound, size=filter_sz).astype(th.config.floatX), borrow=True, name='decoder_weights_' + str(layer_id))
            self._params += [self.w_dec]


    def __call__(self):
        """
        Returns a list of selected parameters
        """
        return self._params


    def export(self, path, which=None):
        """
        Export selected parameters matrices to npy files

        Parameters
        ----------
        which:      string or NoneType
                    if which == encoder, then export only encoder parameters.
                    otherwise, it returns all parameters
        """
        make_dir(path)
        for param in self._params:
            fn = os.path.join(path, "{}.npy".format(str(param)))
            np.save(fn, param.get_value(borrow=True))


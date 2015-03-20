# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 16:44:15 2014

@author: francis
"""

if __name__ == '__main__':
    import sys
    sys.path.append("../../")

import os
import theano                       as th
import numpy                        as np
import theano.tensor                as T

from math                           import sqrt
from numpy.random                   import uniform
from learn.utils.corrupt            import corrupt
from utils.files.path               import make_dir
from utils.dict.dd                  import dd
from theano.tensor                  import eq
from learn.operators.batchdot       import batchdot



class maxout(object):

    def __init__(self, layer_inp, params):

        # Layer input
        self.layer_inp = layer_inp

        # Model parameters
        self.params = params


    @property
    def output(self):
        scale = 1-self.params.hp.inp_corruption_level if self.params.hp.inp_corruption_type == 'zeromask' else 1.
        return self.enc(self.layer_inp).max(axis=2)*scale


    def learn(self, trainer=None):

        # Initialize learning
        if trainer:

            # For readability
            inp_cor_level = self.params.hp.inp_corruption_level
            inp_cor_type  = self.params.hp.inp_corruption_type
            hid_cor_level = self.params.hp.hid_corruption_level
            hid_cor_type  = self.params.hp.hid_corruption_type
            x = self.layer_inp

            # Non-denoising autoencoder input and hidden unit scale
            inp_scale = 1-inp_cor_level if inp_cor_type == 'zeromask' else 1.
            hid_scale = 1-hid_cor_level if hid_cor_type == 'zeromask' else 1.

            # Non-denoising autoencoder cost (validation cost)
            clean_cost = self.__cost(x, self.__dec(self.__enc(x))*inp_scale*hid_scale)

            # Denoising autoencoder cost (test cost)
            h = self.__enc(x, inp_cor_type, inp_cor_level)
            v = self.__dec(h, hid_cor_type, hid_cor_level)
            noisy_cost = self.__cost(x, v)

            # Set trainer with model inputs, parameters and outputs
            self.trainer = trainer(None, noisy_cost, None, clean_cost, None, self.params(), self.params.debug_call)

        # Perform learning
        self.trainer.learn()


    def enc(self, x, corruption_type=None, corruption_level=0):

        # For readability
        W = self.params().encoder_weights

        # Corrupt input
        x = corrupt(x, corruption_type, corruption_level)

        # Encode, max and return
        return batchdot(x, W)


    def dec(self, h, corruption_type=None, corruption_level=0):

        # For readability
        W = self.params().encoder_weights

        # Corrupt input
        h = corrupt(h, corruption_type, corruption_level)

        # Maxout and decode
        return batchdot(h*eq(h, h.max(0, keepdims=True)), W.dimshuffle(0, 2, 1)).sum(0)


    def cost(self, x, v):
        # Make autoencoder cost function from input and output units
        return ((v-x)**2).sum(axis = 1).mean()


class maxout_params(object):

    def __init__(self, hp, layer_id=''):
        """
        Initializes maxout parameters.

        Parameters
        ----------
        hp          dict like object
                    Contains hyperparameters
                    .nb_inp = number of layer inputs
                    .nb_max = number of inputs per maxout units
                    .nb_out = number of maxout units
                    .inp_corruption_level
                    .inp_corruption_type
                    .hid_corruption_level
                    .hid_corruption_type

        """

        # Format layer_id
        self.layer_id = '_' + str(layer_id)

        # Hyperparameters
        self.hp = hp

        # Set parameter dictionary
        self.__params = [('encoder_weights', None)]
        self.__params = dd(self.__params)


    def __call__(self):
        """
        Utility function that returns the list of parameter shared variables
        """

        hp = self.hp
        layer_id = self.layer_id

        if not self.__params.encoder_weights:
            self.__params.encoder_weights = th.shared(uniform(-sqrt(6./hp.nb_inp), sqrt(6./hp.nb_inp), size=(hp.nb_max, hp.nb_inp, hp.nb_out)).astype(th.config.floatX), name="encoder_weights{}".format(layer_id))

        return self.__params


    def debug_call(self):
        pass


    def export(self, path):
        """
        Export parameters shared variables to npy files.  Files are named
        according of the "name" argument of the shared variable

        Parameters
        ----------
        path:       string
                    path for which to export parameter files
        """
        make_dir(path)
        for param in self.__params.values():
            fn = os.path.join(path, "{}.npy".format(str(param)))
            np.save(fn, param.get_value(borrow=True))


    def fromfile(self, path):
        """
        Import parameters that were exported witht he export function.

        Parameters
        ----------
        path:       string
                    path from which to import parameter files
        """

        for param in self.__params.values:
            fn = os.path.join(path, "{}.npy".format(str(param)))
            param.set_value(np.load(fn))
            print 'Loading parameter {} from file with shape {}'.format(str(param), param.get_value().shape)

if __name__ == '__main__':

    # For testing the maxout model
    x = T.matrix('tensor2', dtype = th.config.floatX)

    hp = dd()
    hp.nb_inp = 5
    hp.nb_max = 3
    hp.nb_out = 7
    hp.inp_corruption_level = 0
    hp.inp_corruption_type = None
    hp.hid_corruption_level = 0
    hp.hid_corruption_type = None

    params = maxout_params(hp)
    layer = maxout(x, params)
    h = layer.enc(x)
    d = layer.dec(h)
    c = layer.cost(d, x)
    enc_fn = th.function(inputs = [x], outputs = h)
    dec_fn = th.function(inputs = [x], outputs = d)
    cost_fn = th.function(inputs = [x], outputs = c)
    
    grad = th.grad(c, x)
    
    inp = np.arange(4*hp.nb_inp).reshape((4,hp.nb_inp)).astype(th.config.floatX)
    print enc_fn(inp).shape
    print dec_fn(inp).shape
    print cost_fn(inp)
    
    
    





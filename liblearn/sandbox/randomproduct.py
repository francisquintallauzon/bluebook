# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 12:39:27 2014

@author: francis
"""


if __name__ == '__main__':
    import sys
    sys.path.append("../../")

import os
import theano                       as th
import numpy                        as np
import theano.tensor                as T

from os.path                        import join
from numpy.random                   import uniform
from learn.utils.corrupt            import corrupt
from utils.files.path               import make_dir
from utils.dict.dd                  import dd
from learn.operators.batchdot       import batchdot
from learn.utils.display            import filterstoimg
from learn.utils.activation         import relu


class randomproduct(object):

    def __init__(self, layer_inp, params, result_path):

        # Layer input
        self.layer_inp = layer_inp

        # Model parameters
        self.params = params

        # Output results path
        self.result_path = result_path


    @property
    def out(self):
        h = self.enc(self.layer_inp)
        g = self.gating(h)
        return h*g


    def learn(self, model_input, trainer=None):

        # Initialize learning
        if trainer:

            # For readability
            inp_cor_level = self.params.hp.inp_corruption_level
            inp_cor_type  = self.params.hp.inp_corruption_type
            hid_cor_level = self.params.hp.hid_corruption_level
            hid_cor_type  = self.params.hp.hid_corruption_type
            x = self.layer_inp

            # Non-denoising autoencoder cost (validation cost)
            h = self.enc(x)
            g = self.gating(h)
            v = self.dec(h*g)
            clean_cost = self.cost(x, v)

            # Denoising autoencoder cost (test cost)
            h = self.enc(x, inp_cor_type, inp_cor_level)
            g = self.gating(h)
            v = self.dec(h*g, hid_cor_type, hid_cor_level)
            noisy_cost = self.cost(x, v)

            # Extract optimizable params
            params = [param for param in self.params().values() if isinstance(param, T.sharedvar.SharedVariable)]


            # Set trainer with model inputs, parameters and outputs
            self.trainer = trainer(model_input, None, noisy_cost, None, clean_cost, None, params, "randomproduct{}".format(self.params.layer_id), self.params.debug_call)

        # Perform learning
        self.trainer.learn()


    def enc(self, x, corruption_type=None, corruption_level=0):

        # For readability
        E = self.params().encoder_weights
        b = self.params().encoder_biases
        inp_act = self.params.hp.inp_act

        # Corrupt input
        x = corrupt(x, corruption_type, corruption_level)

        # Scale if using biased noise
        x = x / T.cast(1-corruption_level, th.config.floatX) if corruption_type == 'zeromask' and corruption_level > 0 else x

        # Encode, max and return
        return inp_act(T.dot(x, E)+b)-b


    def gating(self, x):

        # For readability
        G = self.params().gating_weights
        b = self.params().gating_biases
        gat_act = self.params.hp.gat_act

        # Because log expect values > 0
        x = relu(x) + T.cast(10**-20, x.dtype)

        # Compute gating unit
        return gat_act(T.exp(T.dot(T.log(x), G))-b)+b


    def dec(self, x, corruption_type=None, corruption_level=0):

        # For readability
        D = self.params().decoder_weights

        # Corrupt input
        x = corrupt(x, corruption_type, corruption_level)

        # Scale if using biased noise
        x = x / T.cast(1-corruption_level, th.config.floatX) if corruption_type == 'zeromask' and corruption_level > 0 else x

        # Multiply and decode
        return T.dot(x, D)


    def cost(self, x, v):
        # Make autoencoder cost function from input and output units
        return (0.5*(v-x)**2).mean()



class randomproduct_params(object):

    def __init__(self, hp, layer_id=''):
        """
        Initializes geometric mean layer parameters.

        Parameters
        ----------
        hp:          dict like object
                    Contains hyperparameters
                    .nb_inp = number of layer inputs
                    .nb_geo = number of elements per geometric mean
                    .nb_out = number of output units
                    .inp_corruption_level
                    .inp_corruption_type
                    .hid_corruption_level
                    .hid_corruption_type
                    .debug_path
                    .patch_sz = input patch size (if coming from image), for debug purpose

        """
        self.output_debug = False

        # Format layer_id
        self.layer_id = '_' + str(layer_id)

        # Hyperparameters
        self.hp = hp

        # Set parameter dictionary
        self.__params = dd()
        self.__params.encoder_weights = None
        self.__params.decoder_weights = None
        self.__params.gating_weights = None
        self.__params.encoder_biases = None
        self.__params.gating_biases = None

    def __call__(self):
        """
        Utility function that returns the list of parameter shared variables
        """

        hp = self.hp
        layer_id = self.layer_id

        key = 'encoder_weights'
        if not self.__params[key]:
            name = '{}{}'.format(key, layer_id)
            self.__params[key] = th.shared(uniform(low=-np.sqrt(6./hp.nb_inp), high=np.sqrt(6./hp.nb_inp), size=(hp.nb_inp, hp.nb_out)).astype(th.config.floatX), name=name)

        key = 'encoder_biases'
        if not self.__params[key]:
            name = '{}{}'.format(key, layer_id)
            self.__params[key] = th.shared(np.zeros(hp.nb_out, th.config.floatX), name=name)

        key = 'decoder_weights'
        if not self.__params[key]:
            if self.hp.shared:
                self.__params[key] = self.__params.encoder_weights.T
            else:
                name = '{}{}'.format(key, layer_id)
                self.__params[key] = th.shared(uniform(low=-np.sqrt(6./hp.nb_out), high=np.sqrt(6./hp.nb_out), size=(hp.nb_out, hp.nb_inp)).astype(th.config.floatX), name=name)

        key = 'gating_weights'
        if not self.__params[key]:
            name = '{}{}'.format(key, layer_id)
            self.__params[key] = th.shared(np.random.binomial(1, hp.gating_prob, size=(hp.nb_out, hp.nb_out)).astype(th.config.floatX), name=name)

        key = 'gating_biases'
        if not self.__params[key]:
            name = '{}{}'.format(key, layer_id)
            self.__params[key] = th.shared(np.zeros(hp.nb_out, th.config.floatX), name=name)


        return self.__params


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

    def debug_call(self):

        if self.output_debug:
            # For readability
            W = self.__params.encoder_weights.get_value()
            pca = self.__pca_weights
            patch_sz = self.__patch_sz
            debug_path = self.__debug_path
            prefix = self.__prefix

            make_dir(debug_path)
            filterstoimg(np.dot(pca, W), patch_sz, fn = join(debug_path, '{}{}_encoder_weights.tif'.format(prefix, self.layer_id)))

    def set_debug_info(self, pca_weights, patch_sz, debug_path, prefix):
        make_dir(debug_path)
        self.__pca_weights = pca_weights
        self.__patch_sz = patch_sz
        self.__debug_path = debug_path
        self.__prefix = '' if prefix == None else prefix
        self.output_debug = True

if __name__ == '__main__':

    # For testing the maxout model
    x = T.matrix('tensor2', dtype = th.config.floatX)

    hp = dd()
    hp.nb_inp = 5
    hp.nb_geo = 3
    hp.nb_out = 7
    hp.inp_corruption_level = 0
    hp.inp_corruption_type = None
    hp.hid_corruption_level = 0
    hp.hid_corruption_type = None

    params = geometric_params(hp)
    layer = geometric(x, params)
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









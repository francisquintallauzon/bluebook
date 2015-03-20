# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 22:52:17 2013

@author: francis
"""

import os
import theano                       as th
import numpy                        as np
import theano.tensor                as T

from numpy.random                   import uniform
from learn.utils.corrupt            import corrupt
from learn.train.sgd                import sgd
from utils.files.path               import make_dir
from learn.utils.display            import filterstoimg


class dae(object):
    def __init__(self, params, act, model_inp, layer_inp, inp_corruption_type = None, inp_corruption_level = 0, hid_corruption_type = None, hid_corruption_level = 0, L1_hiddens = 0, L2_weights = 0):
        """
        Denoising autoencoder
        """

        ###########################################################################################
        # Model
        self.act = act
        self.params = params
        self.model_inp = model_inp
        self.layer_inp = layer_inp
        self.inp_corruption_type = inp_corruption_type
        self.inp_corruption_level = inp_corruption_level
        self.hid_corruption_type = hid_corruption_type
        self.hid_corruption_level = hid_corruption_level
        self.L1_hiddens = L1_hiddens
        self.L2_weights = L2_weights
        self.debug = False

        # Model feature layer output
        self.out, _ = self._model(layer_inp, inp_corruption_type, inp_corruption_level, hid_corruption_type, hid_corruption_level, L1_hiddens, L2_weights)

    def learn(self, lookback, max_epoch, minibatch_sz, train_batch_size_ratio, valid_batch_size_ratio, learning_rate, momentum, fetcher, fetchonce=True, maxnorm=0, do_validation=True):

        # Non-denoising version of the autoencoder (for measuring validation cost)
        inp_scale = 1-self.inp_corruption_level if self.inp_corruption_type == 'zeromask' else 1.
        _, clean_cost = self._model(self.layer_inp*inp_scale)

        # Noisy version of the gated ae
        _, noisy_cost = self._model(self.layer_inp, self.inp_corruption_type, self.inp_corruption_level, self.hid_corruption_type, self.hid_corruption_level, self.L1_hiddens, self.L2_weights)

        # Initialize learning
        self.trainer = sgd(self.model_inp, None, noisy_cost, None, clean_cost, None, self.params(), maxnorm, self.params._debug_call)

        # Learn
        self.trainer.learn(lookback, max_epoch, minibatch_sz, train_batch_size_ratio, valid_batch_size_ratio, learning_rate, momentum, fetcher, fetchonce, do_validation)
        for i in range(6):
            learning_rate = self.trainer._learning_rate / 2
            lookback *= 1.25
            self.trainer.learn(lookback, max_epoch, minibatch_sz, train_batch_size_ratio, valid_batch_size_ratio, learning_rate, momentum, fetcher, fetchonce, do_validation)


    def _model(self, layer_inp, inp_corruption_type=None, inp_corruption_level=0, hid_corruption_type = None, hid_corruption_level = 0, L1_hiddens = 0, L2_weights = 0):

        # Make model
        act = self.act
        inp = corrupt(layer_inp, inp_corruption_type, inp_corruption_level)
        hid = act(T.dot(inp, self.params.w_enc) + self.params.b)
        hid = corrupt(hid, hid_corruption_type, hid_corruption_level)
        out = T.dot(hid, self.params.w_dec) 

        # Make cost function
        cost = T.mean((0.5*(out - layer_inp)**2).sum(axis = 1))

        # Add L1 hiddens cost
        if L1_hiddens > 0:
            cost += L1_hiddens * abs(hid).sum(1).mean()

        # Add L2 weight cost 
        if L2_weights > 0:
            cost += L2_weights * ((self.params.w_enc**2.).sum() + 
                                  (self.params.w_dec**2.).sum())

        return hid, cost


    def export(self, path):
        make_dir(path)
        for param in self.encoder_params:
            fn = os.path.join(path, "{}.npy".format(str(param)))
            np.save(fn, param.get_value(borrow=True))


    def clear(self):
        if hasattr(self, 'trainer'):
            del self.trainer


class dae_params(object):
    def __init__(self, act, nb_inp, nb_hid, layer_ind = 0, tied_weights = False):
        """
        Initializes gated dae parameters
        """
        
        # Format layer_id  
        layer_ind = '' if layer_ind == None else '_' + str(layer_ind)        
        
        # Hyperparameters
        self.nb_inp = nb_inp
        self.nb_hid = nb_hid
        self.act = act
        self.tied_weights = tied_weights
        
        # Parameters
        self._params = []

        # Parameters
        self.b = th.shared(np.zeros(nb_hid, dtype = th.config.floatX), borrow=True, name='hidden_unit_biases{}'.format(layer_ind))
        self._params = [self.b]

        factor = 4. if act == T.nnet.sigmoid else 1.
        self.w_enc = th.shared(uniform(low=-factor*np.sqrt(6./nb_inp), high=factor*np.sqrt(6./nb_inp), size=(nb_inp, nb_hid)).astype(th.config.floatX), borrow=True, name='encoder_weights{}'.format(layer_ind))
        self._params += [self.w_enc]

        if tied_weights:
            self.w_dec = self.w_enc.T
        else:
            self.w_dec = th.shared(uniform(low=-factor*np.sqrt(6./nb_hid), high=factor*np.sqrt(6./nb_hid), size=(nb_hid, nb_inp)).astype(th.config.floatX), borrow=True, name='decoder_weights{}'.format(layer_ind))
            self._params += [self.w_dec]


    def __call__(self):
        """
        Utility function that returns the list of parameter shared variables
        """   
        return self._params  


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
        for param in self._params:
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
        if path == None:
            return
        for param in self._params:
            try:
                fn = os.path.join(path, "{}.npy".format(str(param)))
                val = np.load(fn)
                print 'Loading parameter {} from file with shape {}'.format(str(param), val.shape)
                param.set_value(val)
            except:
                print 'Could not load file for parameter {}'.format(str(param))
                pass
            


    def _set_debug_info(self, pca_weights, patch_sz, debug_path):
        make_dir(debug_path)
        self.pca_weights = pca_weights
        self.debug = True
        self.patch_sz = patch_sz
        self.debug_path = debug_path


    def _debug_call(self):
        if self.debug:
            filterstoimg(np.dot(self.pca_weights, self.params.w_enc.get_value()), self.patch_sz, fn = os.path.join(self.debug_path, 'unsupervised_dae.tif'))

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 17:47:56 2014

@author: francis
"""

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
from learn.train.adadelta           import adadelta
from utils.files.path               import make_dir
from learn.utils.display            import filterstoimg


class multicost_dae(object):
    def __init__(self, params, 
                       act, 
                       model_inp, 
                       layer_inp, 
                       whiten = True,
                       inp_corruption_type = None, 
                       inp_corruption_level = 0, 
                       hid_corruption_type = None, 
                       hid_corruption_level = 0, 
                       cost_dropout_level = 0, 
                       L1_hiddens = 0, 
                       L2_weights = 0):
        """
        Denoising autoencoder
        """

        ###########################################################################################
        # Model
        self.act = act
        self.params = params
        self.model_inp = model_inp
        self.layer_inp = layer_inp
        self.whiten = True
        self.inp_corruption_type = inp_corruption_type
        self.inp_corruption_level = inp_corruption_level
        self.hid_corruption_type = hid_corruption_type
        self.hid_corruption_level = hid_corruption_level
        self.cost_dropout_level = cost_dropout_level
        self.L1_hiddens = L1_hiddens
        self.L2_weights = L2_weights




    def learn(self, lookback, 
                    max_epoch, 
                    minibatch_sz, 
                    train_batch_size_ratio, 
                    valid_batch_size_ratio, 
                    learning_rate,
                    momentum, 
                    fetcher, 
                    fetchonce=False, 
                    maxnorm=0, 
                    do_validation=True):
                        
        # Number of preprocessing 
        nb_minibatch = int(1/train_batch_size_ratio)

        print '    Learn dc centering'
        updates = [(self.params.dc, self.params.dc + self.layer_inp.mean(0) / nb_minibatch)]
        fn = th.function(inputs=[self.model_inp], updates = updates)
        for i in range(nb_minibatch / 10):
            fn(fetcher(batch_size_ratio=train_batch_size_ratio, output_labels=False))
        
        print '    Learn global contrast normalization'
        updates = [(self.params.std, self.params.std + (self.layer_inp**2).mean(0))]
        fn = th.function(inputs=[self.model_inp], updates = updates)
        for i in range(nb_minibatch / 10):
            fn(fetcher(batch_size_ratio=train_batch_size_ratio, output_labels=False))
        th.function(inputs=[], updates = [(self.params.std, T.sqrt(self.params.std / nb_minibatch - self.params.dc**2))])()
            
        print '    Learn covariance'
        cov = th.shared(np.zeros((self.params.nb_inp, self.params.nb_inp), dtype=th.config.floatX), name = 'cov_layer')
        mat = (self.layer_inp - self.params.dc) / self.params.std
        updates = [(cov, cov + T.dot(mat.T, mat) / T.cast(nb_minibatch * mat.shape[0], th.config.floatX))]
        fn = th.function(inputs=[self.model_inp], updates=updates)
        for i in range(nb_minibatch / 10):
            fn(fetcher(batch_size_ratio=train_batch_size_ratio, output_labels=False))
        
        print '    Learn pca'
        l, w = np.linalg.eig(cov.get_value())
        
        # Sort eigenvalues        
        sort = np.argsort(l)[::-1][:self.params.nb_pca]
        
        print '    pca with {} components retains {:.1%} of variance'.format(self.params.nb_pca, l[sort].sum() / l.sum())
        
        # Extract most important eigen vectors and whiten if requested by the user            
        if self.whiten:
            w  = w[:, sort] / np.sqrt(l[sort])
        else :
            w = w[:, sort]

        self.params.pca.set_value(w.astype(th.config.floatX))

        # Create a non-denoising version of the autoencoder (for measuring the validation cost)
        inp_scale = 1-self.inp_corruption_level if self.inp_corruption_type == 'zeromask' else 1.
        _, clean_cost = self._model(self.layer_inp*inp_scale)

        # Noisy version of the gated ae
        _, noisy_cost = self._model(self.layer_inp, self.inp_corruption_type, self.inp_corruption_level, self.hid_corruption_type, self.hid_corruption_level, self.L1_hiddens, self.L2_weights)

        # Initialize learning
        self.trainer = adadelta(self.model_inp, None, noisy_cost, None, clean_cost, None, self.params(), maxnorm, self._debug_call)

        # Learn
        self.trainer.learn(lookback, max_epoch, minibatch_sz, train_batch_size_ratio, valid_batch_size_ratio, learning_rate, momentum, fetcher, fetchonce, do_validation)


    @property
    def out(self):
        _out, _ = self._model(self.layer_inp, self.inp_corruption_type, self.inp_corruption_level, self.hid_corruption_type, self.hid_corruption_level, self.cost_dropout_level, self.L1_hiddens, self.L2_weights)
        return _out



    def _model(self, layer_inp, 
                     inp_corruption_type=None, 
                     inp_corruption_level=0, 
                     hid_corruption_type = None, 
                     hid_corruption_level = 0, 
                     cost_dropout_level = 0, 
                     L1_hiddens = 0, 
                     L2_weights = 0):

        # For conciseness
        act = self.act

        # DC centering
        inp = layer_inp - self.params.dc

        # Contrast normalization
        inp /= self.params.std

        # Corrupt input
        enc = corrupt(inp, inp_corruption_type, inp_corruption_level)

        # Apply PCA
        enc = T.dot(enc, self.params.pca)

        # Encode
        enc = act(T.dot(enc, self.params.w_enc) + self.params.b)

        # Corrupt encoder output
        enc = corrupt(enc, hid_corruption_type, hid_corruption_level)

        # Decode
        dec = T.dot(enc, self.params.w_dec) 

        # Reverse PCA
        dec = T.dot(dec, self.params.pca.T)

        # Make cost function
        cost = corrupt(0.5*(dec - inp)**2, 'zeromask', cost_dropout_level)
        cost = T.mean(cost.sum(axis = 1))

        # Add L1 hiddens cost
        if L1_hiddens > 0:
            cost += L1_hiddens * T.sum(abs(enc))

        # Add L2 weight cost 
        if L2_weights > 0:
            cost += L2_weights * ((self.params.w_enc**2.).sum() + 
                                  (self.params.w_dec**2.).sum())

        # Add orthogonality cost
        # dot = T.dot(self.params.w_enc.T, self.params.w_enc)
        # cost += 0.000000000000000001 * T.sum(abs(dot - T.zeros_like(dot)))

        return enc, cost


    def _set_debug_info(self, patch_sz, debug_path):
        make_dir(debug_path)
        self.debug_enabled = True
        self.patch_sz = patch_sz
        self.debug_path = debug_path


    def _debug_call(self):
        if hasattr(self, 'debug_enabled'):
            if self.debug_enabled:
                filterstoimg(np.dot(self.params.pca.get_value(), self.params.w_enc.get_value()), self.patch_sz, fn = os.path.join(self.debug_path, 'unsupervised_dae.tif'))


    def export(self, path):
        make_dir(path)
        for param in self.encoder_params:
            fn = os.path.join(path, "{}.npy".format(str(param)))
            np.save(fn, param.get_value(borrow=True))


    def clear(self):
        if hasattr(self, 'trainer'):
            del self.trainer

class multicost_dae_params(object):
    def __init__(self, act, nb_inp, nb_pca, nb_hid, layer_ind = 0, tied_weights = False):
        """
        multicost_dae parameters
        """

        #print 'multicost_dae_params layer {} with nb_inp = {}, nb_pca = {}, nb_hid = {}, tied_weights = {}'.format(layer_ind, nb_inp, nb_pca, nb_hid, tied_weights)
        
        # Hyperparameters
        self.nb_inp = nb_inp
        self.nb_pca = nb_pca
        self.nb_hid = nb_hid
        self.act = act
        self.tied_weights = tied_weights
        
        # Parameters
        self._params = []

        # Pre-processing paramters (not to be optimized)
        self.dc = th.shared(np.zeros(nb_inp, dtype=th.config.floatX), name = 'dc_layer_{}'.format(layer_ind))
        self._params += [self.dc]

        self.std = th.shared(np.zeros(nb_inp, dtype=th.config.floatX), name = 'std_layer_{}'.format(layer_ind))
        self._params += [self.std]

        self.pca = th.shared(np.zeros((nb_inp, nb_pca), dtype=th.config.floatX), name = 'std_layer_{}'.format(layer_ind))
        self._params += [self.pca]

        # Parameters
        self.b = th.shared(np.zeros(nb_hid, dtype = th.config.floatX), borrow=True, name='hidden_unit_biases_layer_{}'.format(layer_ind))
        self._params = [self.b]

        factor = 4. if act == T.nnet.sigmoid else 1.
        self.w_enc = th.shared(uniform(low=-factor*np.sqrt(6./nb_pca), high=factor*np.sqrt(6./nb_pca), size=(nb_pca, nb_hid)).astype(th.config.floatX), borrow=True, name='encoder_weights_layer_{}'.format(layer_ind))
        self._params += [self.w_enc]

        if tied_weights:
            self.w_dec = self.w_enc.T
        else:
            self.w_dec = th.shared(uniform(low=-factor*np.sqrt(6./nb_hid), high=factor*np.sqrt(6./nb_hid), size=(nb_hid, nb_pca)).astype(th.config.floatX), borrow=True, name='decoder_weights_layer_{}'.format(layer_ind))
            self._params += [self.w_dec]


    def __call__(self):
        """
        Utility function that returns the list of parameter shared variables to be optimized
        """
        if self.tied_weights:
            return[self.b, self.w_enc]
        else:
            return[self.b, self.w_enc, self.w_dec]


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
            


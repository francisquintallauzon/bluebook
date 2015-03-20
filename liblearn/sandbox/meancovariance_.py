# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 22:52:17 2013

@author: francis
"""

import theano           as th
import numpy            as np
import theano.tensor    as T

from os.path            import join
from numpy.random       import uniform
from learn.utils        import corrupt
from learn.train        import sgd
from utils.path         import make_dir
from learn.utils        import filterstoimg
from learn.layer        import base
from layer.utils        import shared_x
from math import sqrt


class meancovariance(base):
    def __init__(self, dae_act, gdae_act, nb_inp, nb_hid, nb_fac, nb_map, shared, divide, layer_id='',
                       W=None, B=None, FACX=None, FACY=None, M_enc=None, M_dec=None, B_map=None, B_dec=None):
        base.__init__(self, layer_id, W=W, B=B, FACX=FACX, FACY=FACY, M_enc=M_enc, B_map=B_map, B_dec=B_dec)

        # Hyperparameters
        self.nb_inp = nb_inp
        self.nb_hid = nb_hid
        self.nb_fac = nb_fac
        self.nb_map = nb_map
        self.shared = shared
        self.divide = divide
        self.dae_act = dae_act
        self.gdae_act = gdae_act

        # Pre-calculation for conciseness
        daef = 4. if dae_act == T.nnet.sigmoid else 1.
        gdaef = 4. if gdae_act == T.nnet.sigmoid else 1.

        if W == None and nb_hid > 0:
            self.W = shared_x(uniform(-daef*sqrt(6./nb_inp), daef*sqrt(6./nb_inp), size=(nb_inp, nb_hid)), name='W{}'.format(layer_id))

        if B == None and nb_hid > 0:
            self.B = shared_x(np.zeros(nb_hid), name='B{}'.format(layer_id))

        if FACX is None and nb_fac > 0:
            self.FAC_X = shared_x(uniform(-gdaef*sqrt(6./(nb_inp+nb_fac)), gdaef*sqrt(6./(nb_inp+nb_fac)), size=(nb_inp, nb_fac)), name='FACX{}'.format(layer_id))

        if FACY is None and nb_fac > 0:
            self.FAC_Y = self.FAC_X if shared else shared_x(uniform(-gdaef*sqrt(6./(nb_inp+nb_fac)), gdaef*sqrt(6./(nb_inp+nb_fac)), size=(nb_inp, nb_fac)), name='FACX{}'.format(layer_id))

        if M_enc is None and nb_fac > 0 and nb_map > 0:
            self.M_enc = shared_x(uniform(-gdaef*sqrt(6./(nb_fac+nb_map)), gdaef*sqrt(6./(nb_fac+nb_map)), size=(nb_fac, nb_map)), name='M_enc{}'.format(layer_id))
            self.M_dec = shared_x(uniform(-gdaef*sqrt(6./(nb_fac+nb_map)), gdaef*sqrt(6./(nb_fac+nb_map)), size=(nb_map, nb_fac)), name='M_dec{}'.format(layer_id))
            self.BMAP = th.shared(np.zeros(nb_map, dtype = th.config.floatX), borrow=True, name='mapping_unit_biases{}'.format(layer_id))

        self.BVIS = th.shared(np.zeros(nb_inp, dtype = th.config.floatX), borrow=True, name='decoder_unit_biases{}'.format(layer_id))




    def learn(self, corruption_type= None, corruption_level = 0, L1_hiddens = 0, L2_weights = 0):

        # Non-denoising version of the autoencoder (for measuring validation cost)
        inp_scale = 1-self.corruption_level if self.corruption_type == 'zeromask' else 1.
        _, clean_cost = self._model(self.layer_inp*inp_scale)

        # Noisy version of the gated ae
        _, noisy_cost = self._model(self.layer_inp, self.corruption_type, self.corruption_level, self.L1_hiddens, self.L2_weights)

        # Initialize learning
        self.trainer = sgd(self.model_inp, None, noisy_cost, None, clean_cost, None, self.params(), maxnorm, self.params.debug_call)

        # Learn
        self.trainer.learn(lookback, max_epoch, minibatch_sz, train_batch_size_ratio, valid_batch_size_ratio, learning_rate, momentum, fetcher, fetchonce, do_validation, decay_rate)


    def _model(self, layer_inp, corruption_type=None, corruption_level=0, L1_hiddens = 0, L2_weights = 0):

        # Make model
        dae_act = self.dae_act
        gdae_act = self.gdae_act
        inp = corrupt(layer_inp, corruption_type, corruption_level)

        vis_gated = 0
        if self.params.nb_map and self.params.nb_fac:
            if self.params.shared:
                fac_enc = T.dot(inp, self.params.FAC)
                mapping = gdae_act(T.dot(fac_enc**2, self.params.MAP_enc) + self.params.BMAP)
                fac_dec = T.dot(mapping, self.params.MAP_dec)
                if self.divide :
                    vis_gated = T.dot(fac_dec / fac_enc, self.params.FAC.T)
                else:
                    vis_gated = T.dot(fac_dec * fac_enc, self.params.FAC.T)
            else :
                fac_enc_x = T.dot(inp, self.params.FAC_X)
                fac_enc_y = T.dot(inp, self.params.FAC_Y)
                mapping = gdae_act(T.dot(fac_enc_x * fac_enc_y, self.params.MAP_enc) + self.params.BMAP)
                fac_dec = T.dot(mapping, self.params.MAP_dec)
                if self.divide :
                    vis_gated = 0.5 * T.dot(fac_dec / fac_enc_x, self.params.FAC_Y.T) + 0.5 * T.dot(fac_dec / fac_enc_y, self.params.FAC_X.T)
                else:
                    vis_gated = 0.5 * T.dot(fac_dec * fac_enc_x, self.params.FAC_Y.T) + 0.5 * T.dot(fac_dec * fac_enc_y, self.params.FAC_X.T)

        vis_dae = 0
        if self.params.nb_hid:
            hid = dae_act(T.dot(inp, self.params.W) + self.params.BHID)
            vis_dae = T.dot(hid, self.params.W.T)

        # Visible units
        vis = vis_gated + vis_dae + self.params.BVIS

        # Hidden units
        if self.params.nb_map and self.params.nb_fac and self.params.nb_hid:
            out = T.concatenate((mapping, hid), 1)
        elif self.params.nb_map and self.params.nb_fac :
            out = mapping
        else:
            out = hid

        # Make cost function
        cost = T.mean((0.5*(vis - layer_inp)**2).sum(axis = 1))

        # Add L1 hiddens cost
        if L1_hiddens > 0:
            cost += L1_hiddens * (T.sum(abs(mapping)) + T.sum(abs(hid)))

        # Add L2 weight cost
        if L2_weights > 0 and self.params.nb_hid:
            cost += L2_weights * (self.params.W**2.).sum()

        if L2_weights > 0 and self.params.nb_map and self.params.nb_fac:
            cost += L2_weights * ((self.params.FAC**2.).sum() + \
                                  (self.params.MAP_enc**2.).sum() + \
                                  (self.params.MAP_dec**2.).sum())

        return out, cost


    def clear(self):
        if hasattr(self, 'trainer'):
            del self.trainer



class mc_dae_params(object):
    def __init__(self, dae_act, gdae_act, nb_inp, nb_hid, nb_fac, nb_map, shared = True, layer_id=None):
        """
        Initializes gated dae parameters

        Parameters
        ----------
        dae_act :       activation type
                        dae activation object
        gdae_act :      activation type
                        gdae activation object
        nb_inp:         integer type
                        number of input units
        nb_hid:         integer type
                        number of hidden units
        nb_fac:         integer type
                        number of factor units
        nb_map:         integer type
                        number of mapping units
        layer_id        string representable type or NoneType
                        layer identification (useful for debugging)
        """

        # Hyperparameters
        self.nb_inp = nb_inp
        self.nb_hid = nb_hid
        self.nb_fac = nb_fac
        self.nb_map = nb_map
        self.shared = shared

        # Pre-calculation for conciseness
        dae_factor = 4. if dae_act == T.nnet.sigmoid else 1.
        gdae_factor = 4. if gdae_act == T.nnet.sigmoid else 1.

        # Internal list of parameters
        self._params = []

        # Format layer_id
        layer_id = '' if layer_id==None else '_' + str(layer_id)

        if nb_hid > 0:
            self.W = th.shared(uniform(low=-dae_factor*np.sqrt(6./nb_inp), high=dae_factor*np.sqrt(6./nb_inp), size=(nb_inp, nb_hid)).astype(th.config.floatX), borrow=True, name='hidden_unit_weights{}'.format(layer_id))
            self.BHID = th.shared(np.zeros(nb_hid, dtype = th.config.floatX), borrow=True, name='hidden_unit_biases{}'.format(layer_id))
            self._params += [self.W, self.BHID]

        if nb_fac > 0 and nb_map > 0:
            if shared :
                self.FAC = th.shared(uniform(low=-gdae_factor*np.sqrt(6./(nb_inp+nb_fac)), high=gdae_factor*np.sqrt(6./(nb_inp+nb_fac)), size=(nb_inp, nb_fac)).astype(th.config.floatX), borrow=True, name='factor_unit_weights{}'.format(layer_id))
                self._params += [self.FAC]
            else:
                self.FAC_X = th.shared(uniform(low=-gdae_factor*np.sqrt(6./(nb_inp+nb_fac)), high=gdae_factor*np.sqrt(6./(nb_inp+nb_fac)), size=(nb_inp, nb_fac)).astype(th.config.floatX), borrow=True, name='factor_unit_weights_x{}'.format(layer_id))
                self.FAC_Y = th.shared(uniform(low=-gdae_factor*np.sqrt(6./(nb_inp+nb_fac)), high=gdae_factor*np.sqrt(6./(nb_inp+nb_fac)), size=(nb_inp, nb_fac)).astype(th.config.floatX), borrow=True, name='factor_unit_weights_y{}'.format(layer_id))
                self._params += [self.FAC_X, self.FAC_Y]

            self.MAP_enc = th.shared(uniform(low=-gdae_factor*np.sqrt(6./(nb_fac+nb_map)), high=gdae_factor*np.sqrt(6./(nb_fac+nb_map)), size=(nb_fac, nb_map)).astype(th.config.floatX), borrow=True, name='encoder_mapping_units_weights{}'.format(layer_id))
            self.MAP_dec = th.shared(uniform(low=-gdae_factor*np.sqrt(6./(nb_fac+nb_map)), high=gdae_factor*np.sqrt(6./(nb_fac+nb_map)), size=(nb_map, nb_fac)).astype(th.config.floatX), borrow=True, name='decoder_mapping_unit_weights{}'.format(layer_id))
            self.BMAP = th.shared(np.zeros(nb_map, dtype = th.config.floatX), borrow=True, name='mapping_unit_biases{}'.format(layer_id))
            self._params += [self.MAP_enc, self.MAP_dec, self.BMAP]

        self.BVIS = th.shared(np.zeros(nb_inp, dtype = th.config.floatX), borrow=True, name='decoder_unit_biases{}'.format(layer_id))
        self._params += [self.BVIS]


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
            fn = join(path, "{}.npy".format(str(param)))
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

        try:
            # More recent naming
            for param in self._params:
                fn = join(path, "{}.npy".format(str(param)))
                val = np.load(fn)
                print 'Loading parameter {} from file with shape {}'.format(str(param), val.shape)
                param.set_value(val)
        except:
            # Try older naming
            self.W.set_value(np.load(join(path, "Hidden unit X weights.npy")))
            self.BHID.set_value(np.load(join(path, "Hidden unit X biases.npy")))
            self.FAC.set_value(np.load(join(path, "Factor unit X weights.npy")))
            self.MAP_enc.set_value(np.load(join(path, "Encoder mapping unit weights.npy")))
            self.MAP_dec.set_value(np.load(join(path, "Decoder mapping unit weights.npy")))
            self.BMAP.set_value(np.load(join(path, "Mapping unit biases.npy")))
            self.BVIS.set_value(np.load(join(path, "Decoder unit X biases.npy")))


    def set_debug_info(self, pca_weights, patch_sz, debug_path, prefix):
        make_dir(debug_path)
        self.__debug_enabled = True
        self.__pca_weights = pca_weights
        self.__patch_sz = patch_sz
        self.__debug_path = debug_path
        self.__prefix = '' if prefix == None else prefix  + '_'


    def debug_call(self):
        if hasattr(self, '_' + self.__class__.__name__ + '__debug_enabled'):
            if self.__debug_enabled:
                if hasattr(self, 'W'):
                    filterstoimg(np.dot(self.__pca_weights, self.W.get_value()), self.__patch_sz, fn = join(self.__debug_path, self.__prefix + 'w.tif'))

                if hasattr(self, 'FAC'):
                    filterstoimg(np.dot(self.__pca_weights, self.FAC.get_value()), self.__patch_sz, fn = join(self.__debug_path, self.__prefix + 'factors.tif'))

                if hasattr(self, 'FAC_X'):
                    filterstoimg(np.dot(self.__pca_weights, self.FAC_X.get_value()), self.__patch_sz, fn = join(self.__debug_path, self.__prefix + 'factors_x.tif'))

                if hasattr(self, 'FAC_Y'):
                    filterstoimg(np.dot(self.__pca_weights, self.FAC_Y.get_value()), self.__patch_sz, fn = join(self.__debug_path, self.__prefix + 'factors_y.tif'))

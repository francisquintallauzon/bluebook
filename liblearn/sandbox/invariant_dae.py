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
from learn.utils.activation         import relu


def orthomapfrob():
    pass

class invariant_dae(object):
    def __init__(self, params, model_inp, layer_inp, corruption_type= None, corruption_level = 0, dropconnect_level=0, L2_weights = 0, L2_map = 0, frob_orthomap=0):
        
        # Gated dae parameters
        self.params = params
        self.model_inp = model_inp
        self.layer_inp = layer_inp
        self.corruption_type = corruption_type
        self.corruption_level = corruption_level
        self.dropconnect_level = dropconnect_level
        self.L2_weights = L2_weights
        self.L2_map = L2_map
        self.frob_orthomap = frob_orthomap

        # Model feature layer output
        # self.out, _ = self._model(layer_inp, corruption_type, corruption_level, L1_hiddens, L2_weights)

        # Model total number of hidden units (standard hiddens units + mapping units)
        # self.nb_out = self.params.nb_hid + self.params.nb_map


    def learn(self, lookback, max_epoch, minibatch_sz, train_batch_size_ratio, valid_batch_size_ratio, learning_rate, decay_rate, momentum, fetcher, fetchonce=True, maxnorm=0, do_validation=True):

        # Non-denoising version of the autoencoder (for measuring validation cost)
        inp_scale = 1-self.corruption_level if self.corruption_type == 'zeromask' else 1.
        inp_scale = inp_scale * (1-self.dropconnect_level)**2
        _, clean_cost = self._model(self.layer_inp*inp_scale)

        # Noisy version of the gated ae
        _, noisy_cost = self._model(self.layer_inp, self.corruption_type, self.corruption_level, self.dropconnect_level, self.L2_weights, self.L2_map, self.frob_orthomap)

        # Initialize learning
        self.trainer = sgd(self.model_inp, None, noisy_cost, None, clean_cost, None, self.params(), maxnorm, self.params.debug_call)

        # Learn
        self.trainer.learn(lookback, max_epoch, minibatch_sz, train_batch_size_ratio, valid_batch_size_ratio, learning_rate, momentum, fetcher, fetchonce, do_validation, decay_rate)


    def _model(self, layer_inp, corruption_type=None, corruption_level=0, dropconnect_level=0, L2_weights = 0, L2_map = 0, frob_orthomap = 0):
        
        # Params
        p = self.params

        # Corrupt input
        inp = corrupt(layer_inp, corruption_type, corruption_level)

        # Autoencoder model      
        vis = 0
        for map in p.map:
            dc_map = corrupt(map, 'zeromask', dropconnect_level)
            enc = relu(T.dot(T.dot(inp, dc_map), p.w_enc) + p.b_enc)
            dec = T.dot(T.dot(enc, p.w_dec) + p.b_dec, dc_map.T)
            vis += dec 
        
        # Make cost function
        cost = T.mean((0.5*(vis - layer_inp)**2).sum(axis = 1))

        # Add L2 weight cost
        if L2_weights > 0:
            cost += L2_weights * (p.w_enc**2.).sum()
            
        # Add L2 weight cost
        if L2_map > 0:
            for m in p.map:
                cost += L2_map * (m**2.).sum()            
            
        # Add Frobenius mapping unit orthogonality cost
        if frob_orthomap > 0:
            for x, map_x in enumerate(p.map):
                for y, map_y in enumerate(p.map):
                    if x != y:
                        cost += frob_orthomap * abs(T.dot(map_x.T, map_y)).sum()

        return 0, cost



    def clear(self):
        if hasattr(self, 'trainer'):
            del self.trainer
        


class invariant_dae_params(object):
    def __init__(self, nb_inp, nb_hid, nb_map, layer_id=None):
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
        nb_map:         integer type
                        number of mapping units
        layer_id        string representable type or NoneType
                        layer identification (useful for debugging)                        
        """
        
        # Hyperparameters
        self.nb_inp = nb_inp
        self.nb_hid = nb_hid
        self.nb_map = nb_map

        # Internal list of parameters
        self._params = []
        
        # Format layer_id     
        layer_id = '' if layer_id==None else '_' + str(layer_id)
        
        # Encoder weights
        self.w_enc = th.shared(uniform(low=-np.sqrt(3./nb_inp), high=np.sqrt(3./nb_inp), size=(nb_inp, nb_hid)).astype(th.config.floatX), name='encoder_weights{}'.format(layer_id))
        self._params += [self.w_enc]
        
        # Encoder biasses
        self.b_enc = th.shared(np.zeros(nb_hid, dtype = th.config.floatX), name='encoder_bias{}'.format(layer_id))
        self._params += [self.b_enc]

        # Decoder weights
        self.w_dec = th.shared(uniform(low=-np.sqrt(3./nb_hid), high=np.sqrt(3./nb_hid), size=(nb_hid, nb_inp)).astype(th.config.floatX), name='decoder_weights{}'.format(layer_id))
        self._params += [self.w_dec]

        # Decoder biases
        self.b_dec = th.shared(np.zeros(nb_inp, dtype = th.config.floatX), name='decoder_bias{}'.format(layer_id))
        self._params += [self.b_dec]

        # Mapping units
        self.map = []
        for i in range(self.nb_map):
            self.map += [th.shared(uniform(low=-np.sqrt(3./nb_inp), high=np.sqrt(3./nb_inp), size=(nb_inp, nb_inp)).astype(th.config.floatX), name='mapping_{}{}'.format(i, layer_id))]
            self._params += [self.map[i]]



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
            

    def set_debug_info(self, patch_sz, debug_path, prefix):
        make_dir(debug_path)
        self.__debug_enabled = True
        self.__patch_sz = patch_sz
        self.__debug_path = debug_path
        self.__prefix = '' if prefix == None else prefix  + '_'


    def debug_call(self):
        if hasattr(self, '_' + self.__class__.__name__ + '__debug_enabled'):
            if self.__debug_enabled:
                filterstoimg(self.w_enc.get_value(), self.__patch_sz, fn = os.path.join(self.__debug_path, self.__prefix + str(self.w_enc) +".tif"))
                for map in self.map:
                    filterstoimg(np.dot(map.get_value(), self.w_enc.get_value()), self.__patch_sz, fn = os.path.join(self.__debug_path, self.__prefix + str(map) +".tif"))
                
                pw = 10
                size = len(self.map)*pw
                cov = np.zeros((size, size))
                for x, map_x in enumerate(self.map):
                    for y, map_y in enumerate(self.map):
                        cov[y*pw:(y+1)*pw, x*pw:(x+1)*pw] = abs(np.dot(map_x.get_value(), map_y.get_value())).sum()
                
                filterstoimg(cov.flatten()[:, None], (size, size), fn = os.path.join(self.__debug_path, self.__prefix + "mapping_orthogonality" +".tif"))
                

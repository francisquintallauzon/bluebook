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

class topographic_dae(object):
    def __init__(self, model_inp, layer_inp, act, inp_patch_sz, nb_decoders, nb_inp, nb_hid, corruption_type, corruption_level):
        
        ###########################################################################################
        # Model
        self.act = act
        self.nb_inp = nb_inp
        self.nb_hid = nb_hid
        
        # Encoder
        self.W_enc = th.shared(uniform(low=-1./np.sqrt(nb_inp), high=1./np.sqrt(nb_inp), size=(nb_inp, nb_hid)).astype(th.config.floatX), borrow=True, name='Encoder weights')
        self.b_enc = th.shared(np.zeros(nb_hid, dtype = th.config.floatX), borrow=True, name='Encoder biases')

        corr_inp = corrupt(layer_inp, corruption_type, corruption_level)
        noisy_hiddens = act(T.dot(corr_inp, self.W_enc) + self.b_enc)
        clean_hiddens = act(T.dot(layer_inp, self.W) + self.b_enc)

        # Using multiple decoders
        self.W_dec = []
        self.b_dec = []
        self.noisy_cost = []
        self.clean_cost = []

        for dec_ind in range(nb_decoders):
            W_dec = th.shared(uniform(low=-1./nb_hid, high=1./nb_hid, size=(nb_hid, nb_inp)).astype(th.config.floatX), borrow=True, name='Decoder {} weights'.format(dec_ind))
            b_dec = th.shared(np.zeros(nb_inp, dtype = th.config.floatX), borrow=True, name='Decoder {} biases'.format(dec_ind))
            self.W_dec += [W_dec]
            self.b_dec += [b_dec]

            noisy_recons = T.dot(noisy_hiddens, W_dec) + b_dec
            self.noisy_cost += T.mean((0.5*(noisy_recons - layer_inp)**2).sum(axis = 1))
            
            clean_recons = T.dot(clean_hiddens, W_dec) + b_dec
            self.clean_cost += T.mean((0.5*(clean_recons - layer_inp)**2).sum(axis = 1))
            
        # Using multiple masks
        
            
            

        ###########################################################################################
        # For for interactions with other models
        self.inp = layer_inp
        self.out = clean_hiddens
        self.params = [self.W, self.b_enc] + self.W_dec + self.b_dec
        self.encoder_params = [self.W, self.b_enc]
        
        
        ###########################################################################################
        # Trainer object
        self.trainer = sgd(model_inp, self.noisy_cost, self.clean_cost, self.params)
        
    
    def learn(self, lookback, max_epoch, nb_minibatch, batch_sz, learning_rate, momentum, fetcher, fetchonce=True):
        self.trainer.learn(lookback, max_epoch, nb_minibatch, batch_sz, learning_rate, momentum, fetcher, fetchonce)

    def export(self, path):
        make_dir(path)
        for param in self.encoder_params:
            fn = os.path.join(path, "{}.npy".format(str(param)))
            np.save(fn, param.get_value(borrow=True))

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 22:52:17 2013

@author: francis
"""

import numpy                        as np
import theano.tensor                as T

from math                           import sqrt
from os.path                        import join
from learn.layer                    import base
from learn.utils                    import corrupt
from learn.utils                    import shared_x
from learn.utils                    import cast_x
from learn.utils                    import filterstoimg
from utils.path                     import make_dir

class opposingbias(base):

    def __init__(self, act, nb_inp, nb_hid, init = 'gaussian', normalize = True, layer_id=None, W=None, b_enc=None, b_opp=None):
        base.__init__(self, layer_id, W=W, b_enc=b_enc, b_opp=b_opp)

        # Hyperparameters
        self.act = act
        self.nb_inp = nb_inp
        self.nb_hid = nb_hid

        # Parameters
        if not self.W:
            if init=='gaussian':
                w = np.random.normal(0, 0.1, (nb_inp, nb_hid))
            elif init=='uniform':
                w = np.random.uniform(-6/sqrt(nb_inp+nb_hid), 6/sqrt(nb_inp+nb_hid), (nb_inp, nb_hid))

            if normalize:
                w /= np.sqrt((w**2).sum(0))[None, :]

            self.W = shared_x(w, name='W'+self.layer_id)

        self.scale = shared_x(1, name = 'scale'+self.layer_id)

        if not self.b_enc:
            self.b_enc = shared_x(np.zeros(nb_hid), name='b_enc'+self.layer_id)

        if not self.b_opp:
            self.b_opp = shared_x(np.zeros(nb_hid), name='b_opp'+self.layer_id)

        # For debug
        self.__init_W = shared_x(self.W.get_value())


    def __call__(self, inp, corruption_type=None, corruption_level=0):
        return self.act(T.dot(corrupt(inp, corruption_type, corruption_level), self.W*self.scale) + self.b_enc)

    def dec(self, hid, corruption_type=None, corruption_level=0):
        return T.dot(corrupt(hid+self.b_opp, corruption_type, corruption_level), self.W.T*self.scale)

    def cost(self, inp, out):
        return cast_x(0.5)*((out-inp)**2).mean()


    def learn(self, x,  trainer=None, inp_corruption_type=None, inp_corruption_level=0, hid_corruption_type=None, hid_corruption_level=0, learn_scale_first=False, **debug_args):

        # Initialize learning
        if trainer:

            # Non-denoising autoencoder cost (validation cost)
            h = self(x)
            v = self.dec(h)
            clean_cost = self.cost(x, v)

            # Denoising autoencoder cost (test cost)
            h = self(x, inp_corruption_type, inp_corruption_level)
            v = self.dec(h, hid_corruption_type, hid_corruption_level)
            noisy_cost = self.cost(x, v)

            if learn_scale_first:
                # Quick training for weight scaling
                lookback = trainer.lookback
                momentum = trainer.momentum
                trainer.lookback = 5
                trainer.momentum = 0
                trainer([self.scale], noisy_cost, clean_cost, model_id="hidden{}_scaling".format(self.layer_id)).learn()
                trainer.lookback = lookback
                trainer.momentum = momentum

            debug_nodes = {'unsupervised_'+self.model_id+'_h':h}

            # Set trainer with model inputs, parameters and outputs
            self.trainer = trainer(self.params.values(), noisy_cost, clean_cost, model_id="hidden{}".format(self.layer_id), debug_nodes=debug_nodes, debug_calls=(self.debug_call, debug_args))

        # Perform learning
        self.trainer.learn()


    def debug_call(self, debug_path=None, patch_sz=None, pca=None, prefix=''):

        if not debug_path:
            return

        # Make output directory
        make_dir(debug_path)

        # Convert weight matrix into filter imagelets and output to file
        if patch_sz:
            W = T.dot(pca, self.W).eval() if pca != None else self.W.get_value()
            nb_frames = W.shape[0] / np.prod(patch_sz)
            W = W.transpose(1,0)
            W = W.reshape((W.shape[0], nb_frames, -1))
            W = W.transpose(1, 2, 0)
            for i, w in enumerate(W):
                filterstoimg(w, patch_sz, fn=join(debug_path, '{}_{}_frame_{}.png'.format(prefix, str(self.W), i)))

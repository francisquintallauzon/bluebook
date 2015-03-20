# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 20:58:33 2014

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
from utils.matplotlib               import subplots
from utils.path                     import make_dir

class fixedweights(base):

    def __init__(self, act, nb_inp, nb_hid, init = 'gaussian', normalize = 'True', layer_id=None, scale=None, W=None, b=None, b_dec=None):
        base.__init__(self, layer_id, b=b, b_dec=b_dec, scale=scale)

        # Hyperparameters
        self.act = act
        self.nb_inp = nb_inp
        self.nb_hid = nb_hid

        # Parameters
        if not W:
            if init=='gaussian':
                w = np.random.normal(0, 0.1, (nb_inp, nb_hid))
            elif init=='uniform':
                w = np.random.uniform(-6/sqrt(nb_inp+nb_hid), 6/sqrt(nb_inp+nb_hid), (nb_inp, nb_hid))
            if normalize:
                w /= np.sqrt((w**2).sum(0))[None, :]
            self.W = shared_x(w, name='W'+self.layer_id)

        if not self.scale:
            self.scale = shared_x(1, name = 'scale'+self.layer_id)

        if not self.b:
            self.b = shared_x(np.zeros(nb_hid), name='b'+self.layer_id)

        if not self.b_dec:
            self.b_dec = shared_x(np.zeros(nb_inp), name='b_dec'+self.layer_id)

    def __call__(self, inp, corruption_type=None, corruption_level=0):
        return self.act(T.dot(corrupt(inp, corruption_type, corruption_level), self.W*self.scale) + self.b)

    def dec(self, hid, corruption_type=None, corruption_level=0):
        return T.dot(corrupt(hid, corruption_type, corruption_level), self.W.T*self.scale) + self.b_dec

    def cost(self, inp, out):
        return cast_x(0.5)*((out-inp)**2).mean()


    def learn(self, x,  trainer=None, inp_corruption_type=None, inp_corruption_level=0, hid_corruption_type=None, hid_corruption_level=0, **debug_args):

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

            # Set trainer with model inputs, parameters and outputs
            self.trainer = trainer(self.params.values(), noisy_cost, clean_cost, model_id="hidden{}".format(self.layer_id), debug_calls=(self.debug_call, debug_args))

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


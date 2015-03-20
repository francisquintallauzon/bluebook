# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 13:53:05 2014

@author: francis
"""

if __name__ == '__main__':
    import sys
    sys.path.append("../../")

import theano                       as th
import numpy                        as np
import theano.tensor                as T
from os.path                        import join
from numpy.random                   import uniform
from learn.utils                    import corrupt
from utils.path                     import make_dir
from learn.utils                    import filterstoimg
from learn.utils                    import step
from learn.layer                    import base


class zerobias(base):

    def __init__(self, threshold, nb_inp, nb_hid, layer_id='', W=None):
        base.__init__(self, layer_id, W=W)

        # Hyperparameters
        self.threshold = threshold
        self.nb_inp = nb_inp
        self.nb_hid = nb_hid

        # Set optimizable parameter dictionary
        init = uniform(low=-1, high=1, size=(nb_inp, nb_hid)).astype(th.config.floatX)
        init = init/np.sqrt((init**2).sum(0))[None,:]
        self.W = W if W else th.shared(init, name='W'+self.layer_id)


    def learn(self, x, trainer=None, debug_calls=None, div_factor=0, inp_corruption_type=None, inp_corruption_level=0, hid_corruption_type=None, hid_corruption_level=0):

        # Initialize learning
        if trainer:

            # Non-denoising autoencoder cost (validation cost)
            h = self(x)
            v = self.dec(h)
            clean_cost = self.cost(x, v)

            # Denoising autoencoder cost (test cost)
            h = self(x, inp_corruption_type, inp_corruption_level)
            v = self.dec(h, hid_corruption_type, hid_corruption_level)
            noisy_cost = self.cost(x, v, div_factor)

            # Set trainer with model inputs, parameters and outputs
            self.trainer = trainer(self.params.values(), noisy_cost, clean_cost, model_id="zerobias{}".format(self.layer_id),
                                   debug_calls=debug_calls,
                                   debug_nodes={"unsupervised{}_act".format(self.layer_id):self(x)})

        # Perform learning
        self.trainer.learn()


    def __call__(self, x, corruption_type=None, corruption_level=0):

        # Corrupt input
        x = corrupt(x, corruption_type, corruption_level)

        # Encode
        y = T.dot(x, self.W)

        # Select output
        return step(y-self.threshold) * y


    def reverse(self, y):
        pass

    def dec(self, h, corruption_type=None, corruption_level=0):

        # Corrupt input
        h = corrupt(h, corruption_type, corruption_level)

        # Decode
        return T.dot(h, self.W.T)


    def cost(self, x, v, div_factor=0):

        # Regularizer : filter norms dirvergence cost
        nrm = T.sqrt((self.W**2).sum(0)).dimshuffle(0, 'x')
        div = div_factor * (nrm**2 - 2*T.dot(nrm, nrm.T) + (nrm.T)**2).sum()

        # Reconstruction cost
        cost = T.cast(0.5, x.dtype)*((v-x)**2).sum(1).mean()

        return cost + div


    def debug_call(self, debug_path=None, patch_sz=None, pca=None, prefix=''):

        class functor(object):
            def __init__(self, obj, debug_path=None, patch_sz=None, pca=None, prefix=''):
                self.W = obj.W
                self.layer_id = obj.layer_id
                self.debug_path = join(debug_path, 'filters')
                self.patch_sz = patch_sz
                self.pca = pca
                self.prefix = prefix

                # Make output directory
                make_dir(self.debug_path)

            def __call__(self):
                W = T.dot(pca, self.W).eval() if pca else self.W.get_value()
                nb_frames = W.shape[0] / np.prod(patch_sz)
                W = W.transpose(1,0)
                W = W.reshape((W.shape[0], nb_frames, -1))
                W = W.transpose(1, 2, 0)
                for i, w in enumerate(W):
                    filterstoimg(w, patch_sz, fn=join(self.debug_path, '{}{}_encoder_weights_frame_{}.tif'.format(prefix, self.layer_id, i)))

        return None if debug_path is None else functor(self, debug_path, patch_sz, pca, prefix)

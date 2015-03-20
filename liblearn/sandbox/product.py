# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 22:52:17 2013

@author: francis
"""


if __name__ == '__main__':
    import sys
    sys.path.append("../../")

import os
import theano                       as th
import numpy                        as np
import theano.tensor                as T
import matplotlib.pyplot            as plt
from os.path                        import join
from numpy.random                   import uniform, binomial
from learn.utils.corrupt            import corrupt
from utils.files.path               import make_dir
from utils.dict.dd                  import dd
from learn.utils.display            import filterstoimg
from learn.utils.display            import imsave
from utils.matplotlib.figure        import subplots
from utils.matplotlib.hist          import hist
from learn.utils.activation         import step, relu


class product(object):

    def __init__(self, layer_inp, params, result_path):

        if layer_inp.ndim == 3:
            layer_inp = layer_inp.dimshuffle(1, 0, 2)

        # Layer input
        self.layer_inp = layer_inp

        # Model parameters
        self.params = params

        # Output results path
        self.result_path = result_path

    @property
    def out(self):
        pass
    
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
            v = self.dec(h)
            clean_cost = self.cost(x, v)

            # Denoising autoencoder cost (test cost)
            h = self.enc(x, inp_cor_type, inp_cor_level)
            v = self.dec(h, hid_cor_type, hid_cor_level)
            noisy_cost = self.cost(x, v)

            # Set trainer with model inputs, parameters and outputs
            self.trainer = trainer(model_input, None, noisy_cost, None, clean_cost, None, self.params().values(), "geometric{}".format(self.params.layer_id), self.params.debug_call)

        # Perform learning
        self.trainer.learn()


    def enc(self, x, corruption_type=None, corruption_level=0):

        # For readability
        W = self.params().encoder_weights
        S = self.params.encoder_selection
        threshold = T.cast(self.params.hp.threshold, x.dtype)
        
        # Corrupt input
        x = corrupt(x, corruption_type, corruption_level)

        # Scale if using biased noise
        x = x / T.cast(1-corruption_level, th.config.floatX) if corruption_type == 'zeromask' and corruption_level > 0 else x
        
        # Filters
        f = relu(T.dot(x, W))
        
        # Selection function
        s = step(T.exp(T.dot(T.log(f), S))-threshold)
        
        # Encode
        return s * T.dot(f, S)


    def dec(self, h, corruption_type=None, corruption_level=0):

        # For readability
        W = self.params().encoder_weights
        S = self.params.encoder_selection
        
        # Corrupt input
        h = corrupt(h, corruption_type, corruption_level)

        # Scale if using biased noise
        h = h / T.cast(1-corruption_level, th.config.floatX) if corruption_type == 'zeromask' and corruption_level > 0 else h

        # Decode
        return T.dot(T.dot(h, S.T), W.T)


    def cost(self, x, v):
        # Make autoencoder cost function from input and output units
        return T.cast(0.5, x.dtype)*((v-x)**2).sum(1).mean()



class product_params(object):

    def __init__(self, **kwargs):
        """
        Initializes geometric mean layer parameters.

        Parameters
        ----------
        :          dict like object
                    Contains hyperparameters
        """

        # Set parameter dictionary
        self.__params = dd()
        self.__params.encoder_weights = None

        if 'import_path' in kwargs:
            for param in self.__params.values:
                fn = os.path.join(kwargs.import_path, "{}.npy".format(str(param)))
                param.set_value(np.load(fn))
                print 'Loading parameter {} from file with shape {}'.format(str(param), param.get_value().shape)
        
        # Hyperparameters
        self.hp = dd(kwargs)
    
        # Parameter not to be optimized        
        init = binomial(1, self.hp.selection_prob, size=(self.hp.nb_out, self.hp.nb_out)).astype(th.config.floatX)
        init /= np.sqrt((init**2).sum(0))
        self.encoder_selection = th.shared(init, name='encoder_selection')

        # Format layer_id
        self.layer_id = '_' + str(self.hp.layer_id) if 'layer_id' in self.hp else ''
        
        self.output_debug = False

    def __call__(self):
        """
        Utility function that returns the list of parameter shared variables
        """

        hp = self.hp
        layer_id = self.layer_id

        key = 'encoder_weights'
        if not self.__params[key]:
            name = '{}{}'.format(key, layer_id)
            init = uniform(low=-1, high=1, size=(hp.nb_inp, hp.nb_out)).astype(th.config.floatX)
            init /= np.sqrt((init**2).sum(0))
            self.__params[key] = th.shared(init, name=name)

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


    def debug_call(self):

        if self.output_debug:
            # For readability
            pca = self.__pca_weights
            patch_sz = self.__patch_sz
            debug_path = self.__debug_path
            prefix = self.__prefix

            # Output orthogonality vizualization
            W = self.__params.encoder_weights.get_value()
            wsq = np.dot(W.T, W)
            wsq[wsq>1] = 1 
            wsq[wsq<-1] = -1
            imsave((wsq + 1) / 2, join(debug_path, 'others', '{}{}_encoder_weights_orthogonality.tif'.format(prefix, self.layer_id)))
            
            # Output filters
            W = self.__params.encoder_weights.get_value()
            W = np.dot(pca, W) if pca else W
            nb_frames = W.shape[0] / np.prod(patch_sz)
            W = W.transpose(1,0)
            W = W.reshape((W.shape[0], nb_frames, -1))
            W = W.transpose(1, 2, 0)
            for i, w in enumerate(W):
                filterstoimg(w, patch_sz, fn=join(debug_path, 'filters', '{}{}_encoder_weights_frame_{}.tif'.format(prefix, self.layer_id, i)))
                
            # Output filters
            W = self.__params.encoder_weights.get_value()
            S = self.encoder_selection.get_value()
            W = np.dot(pca, W) if pca else W
            W = np.dot(W, S)
            nb_frames = W.shape[0] / np.prod(patch_sz)
            W = W.transpose(1,0)
            W = W.reshape((W.shape[0], nb_frames, -1))
            W = W.transpose(1, 2, 0)
            for i, w in enumerate(W):
                filterstoimg(w, patch_sz, fn=join(debug_path, 'filters', '{}{}_encoder_selection_frame_{}.tif'.format(prefix, self.layer_id, i)))                
                
            
    def set_debug_info(self, pca_weights, patch_sz, debug_path, prefix):
        make_dir(join(debug_path, 'filters'))
        make_dir(join(debug_path, 'others'))
        self.__pca_weights = pca_weights
        self.__patch_sz = patch_sz
        self.__debug_path = debug_path
        self.__prefix = '' if prefix == None else prefix
        self.output_debug = True


if __name__ == '__main__':
    
    # For testing the maxout model
    x = T.matrix('tensor2', dtype = th.config.floatX)

    patch_sz = (5,5)
    seq_sz = 5
    hp = dd()
    hp.threshold = 0
    hp.selection_prob = seq_sz
    hp.nb_inp = int(np.prod(patch_sz))*seq_sz
    hp.nb_out = hp.nb_inp
    hp.inp_corruption_level = 0
    hp.inp_corruption_type = None
    hp.hid_corruption_level = 0
    hp.hid_corruption_type = None
    hp.threshold = 1

    params = product_params(**hp)
    params()
    params.set_debug_info(None, patch_sz, './debug', 'test')
    params.debug_call()
    
    
    layer = product(x, params, './results')
    h = layer.enc(x)
    d = layer.dec(h)
    c = layer.cost(d, x)
    enc_fn = th.function(inputs = [x], outputs = h)
    dec_fn = th.function(inputs = [x], outputs = d)
    cost_fn = th.function(inputs = [x], outputs = c)

    grad = th.grad(c, x)
    
    nb_examples = 4
    inp = np.arange(nb_examples*hp.nb_inp).reshape((nb_examples,hp.nb_inp)).astype(th.config.floatX)
    print enc_fn(inp).shape
    print dec_fn(inp).shape
    print cost_fn(inp)



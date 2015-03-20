# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 16:44:15 2014
sigmoid
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
from math                           import sqrt
from numpy.random                   import uniform
from learn.utils.corrupt            import corrupt
from utils.files.path               import make_dir
from utils.dict.dd                  import dd
from learn.operators.batchdot       import batchdot
from learn.utils.display            import filterstoimg
from utils.matplotlib.figure        import subplots
from utils.matplotlib.hist          import hist
from learn.utils.activation         import relu, tanh, step, sigmoid



def __prod(i, x, y):
    sel = (T.neq(T.arange(y.shape[0]), i)).nonzero()
    prod = x * T.prod(y[sel], 0)
    return prod

def mult_by_prod_of_others(x, y):
    if x.ndim == 2 and y.ndim == 3:
        result, _ = th.scan(__prod, outputs_info=None, sequences=[T.arange(y.shape[0])], non_sequences=[x, y])
    else:
        raise NotImplementedError, "x.ndim (={}) must be 2 and y.ndim (={}) must be 3".format(x.ndim, y.ndim)
    return result

class geometric(object):

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
        inp_act = self.params.hp.inp_act
        h = self.enc(self.layer_inp)
        return (inp_act(h)).prod(0)

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
        b = self.params().selection_biases
        f = self.params().selection_factor
        inp_act = self.params.hp.inp_act
        #sel_act = self.params.hp.sel_act
        variant = self.params.hp.variant
        
        # Corrupt input
        x = corrupt(x, corruption_type, corruption_level)

        # Scale if using biased noise
        x = x / T.cast(1-corruption_level, th.config.floatX) if corruption_type == 'zeromask' and corruption_level > 0 else x
        
        # Compute input activation
        x = inp_act(batchdot(x, W))
        
        if variant == 'sigmoid_product':
            x = sigmoid(x.prod(0)+b) * x
        elif variant == 'tanh_product':
            x = tanh(x.prod(0)*f) * x
        elif variant == 'step_product':
            x = step(x.prod(0)-0.01) * x
        elif variant == 'step_sum':
            x = step(x.sum(0)-1) * x
        
        # Encode
        return x


    def dec(self, h, corruption_type=None, corruption_level=0):

        # For readability
        W = self.params().encoder_weights

        # Corrupt input
        h = corrupt(h, corruption_type, corruption_level)

        # Scale if using biased noise
        h = h / T.cast(1-corruption_level, th.config.floatX) if corruption_type == 'zeromask' and corruption_level > 0 else h

        # Decode
        return batchdot(h, W.dimshuffle(0, 2, 1))


    def cost(self, x, v):
        # Make autoencoder cost function from input and output units
        return T.cast(0.5, x.dtype)*((v-x)**2).sum(2).sum(0).mean()



class geometric_params(object):

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
        self.__params.selection_biases = None
        self.__params.selection_factor = None


    def __call__(self):
        """
        Utility function that returns the list of parameter shared variables
        """

        hp = self.hp
        layer_id = self.layer_id

        key = 'encoder_weights'
        if not self.__params[key]:
            name = '{}{}'.format(key, layer_id)
            init = uniform(low=-1, high=1, size=(hp.nb_geo, hp.nb_inp, hp.nb_out))
            init /= np.sqrt((init**2).sum(1))[:, None, :]
            init = init.astype(th.config.floatX)
            self.__params[key] = th.shared(init, name=name)

        key = 'selection_biases'
        if not self.__params[key]:
            name = '{}{}'.format(key, layer_id)
            self.__params[key] = th.shared(np.zeros(hp.nb_out, th.config.floatX)-3, name=name)
            
        key = 'selection_factor'
        if not self.__params[key]:
            name = '{}{}'.format(key, layer_id)
            self.__params[key] = th.shared(np.zeros(hp.nb_out, th.config.floatX)+10, name=name)

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

            for i, w in enumerate(W):
                if pca :
                    filterstoimg(np.dot(pca, w), patch_sz, fn = join(debug_path, 'filters', '{}{}_encoder_weights_{}.tif'.format(prefix, self.layer_id, i)))
                else :
                    filterstoimg(w, patch_sz, fn = join(debug_path, 'filters', '{}{}_encoder_weights_{}.tif'.format(prefix, self.layer_id, i)))
                    
            # Debug output on biases
            fn = join(debug_path, 'histograms', '{}{}_selection_biases.tif'.format(prefix, self.layer_id))
            b = self.__params.selection_biases.get_value()    
            sp = subplots(1, 1, 4, 4, axison=True)
            hist(sp.ax[0,0], b, nbins=20, title='{}{}_selection_biases'.format(prefix, self.layer_id), xlabel='bias value', ylabel='count')
            sp.save(fn, dpi = 150)
            sp.close()
            
            # Debug output on biases
            fn = join(debug_path, 'histograms', '{}{}_selection_factor.tif'.format(prefix, self.layer_id))
            b = self.__params.selection_factor.get_value()    
            sp = subplots(1, 1, 4, 4, axison=True)
            hist(sp.ax[0,0], b, nbins=20, title='{}{}_selection_factor'.format(prefix, self.layer_id), xlabel='bias value', ylabel='count')
            sp.save(fn, dpi = 150)
            sp.close()            

    def set_debug_info(self, pca_weights, patch_sz, debug_path, prefix):
        make_dir(join(debug_path, 'filters'))
        make_dir(join(debug_path, 'histograms'))
        self.__pca_weights = pca_weights
        self.__patch_sz = patch_sz
        self.__debug_path = debug_path
        self.__prefix = '' if prefix == None else prefix
        self.output_debug = True

if __name__ == '__main__':
    
    from learn.utils.activation          import relu, sigmoid, linear

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

    hp = dd()
    hp.type = 'geometric'
    hp.nb_inp = 25
    hp.nb_geo = 3
    hp.nb_out = 7
    hp.variant = 'product'
    hp.inp_act = relu if hp.variant == 'product' else linear
    hp.sel_act = sigmoid 
    hp.inp_corruption_level = 0.5
    hp.inp_corruption_type = np.random.permutation(['zeromask', 'gaussian', None])[0]
    hp.hid_corruption_level = None #0.5
    hp.hid_corruption_type = 0 #np.random.permutation(['zeromask', 'gaussian'])[0]

    params = geometric_params(hp)
    params()
    params.set_debug_info(None, (5,5), './debug', 'test')
    params.debug_call()
    
    
    
    layer = geometric(x, params, './results')
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






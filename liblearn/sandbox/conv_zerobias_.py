# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 11:49:59 2014

@author: francis
"""

if __name__ == '__main__':
    import sys
    sys.path.append("../../")

import theano                       as th
import theano.tensor                as T
import numpy                        as np
from os.path                        import join
from theano.tensor.nnet.conv        import conv2d as conv
from numpy.random                   import uniform
from liblearn.utils           import corrupt
from liblearn.utils           import step
from liblearn.layer           import base
from liblearn.utils           import convfilterstoimg
from libutils.path            import make_dir
from libutils.dict            import dd
from liblearn.utils           import shared_x
from liblearn.utils           import cast_x
from math import sqrt, ceil

class conv_zerobias(base):

    def __init__(self, filter_sz, threshold, init_distribution = 'gaussian', normalize = True, subsample_sz = (1,1), layer_id='', W = None, scale=None):
        """
        Zerobias convolutional layer.

        Parameters
        ----------
        filter_sz           4-dimension tuple of integers
                            filter size (nb filters, stack size, nb lines, nb columns)

        subsample_sz        2-dimension tuple of integers
                            Subsampling factors for in y and x direction

        threshold           numeric type
                            selection function's threshold

        layer_id            str, integer or None
                            layer identifier, helps for debug

        normalize_input     bool
                            If true, normalize input using L2 norm

        W                   th.shared or None
                            convnet's filter weights
        """
        base.__init__(self, layer_id, W=W)

        # Model parameters
        self.threshold = T.cast(threshold, th.config.floatX)
        self.subsample_sz = subsample_sz

        # Initialize optimizable parameters
        if not W:
            if init_distribution == 'uniform':
                w = uniform(low=-1/(3*sqrt(filter_sz[0])), high=1/(3*sqrt(filter_sz[0])), size=filter_sz)
            elif init_distribution == 'gaussian':
                w = np.random.normal(0, 0.01, size=filter_sz)
            else:
                raise NotImplementedError

            if normalize :
                w /= np.sqrt((w**2).sum((1,2,3)))[:,None,None,None]

            self.W = shared_x(w, name='W'+self.layer_id)

        self.scale = shared_x(scale, name = 'scale'+self.layer_id) if scale else shared_x(1, name = 'scale'+self.layer_id)

        # For debug
        self.__init_W = shared_x(self.W.get_value())


    @classmethod
    def output_sz(cls, input_sz, filter_sz):
        return (input_sz[0], filter_sz[0], input_sz[2]-filter_sz[2]+1, input_sz[3]-filter_sz[3]+1)


    def __call__(self, inp, corruption_type=None, corruption_level=0, border_mode = 'valid'):

        # Corrupt input
        inp = corrupt(inp, corruption_type, corruption_level)

        # Scale if using biased noise
        inp = inp / T.cast(1-corruption_level, th.config.floatX) if corruption_type == 'zeromask' and corruption_level > 0 else inp

        # Filtered output
        f = conv(inp, self.W, border_mode=border_mode) * self.scale

        return step(f-self.threshold) * f
        #return step(f/T.sqrt((f**2).sum(1,keepdims=True))-self.threshold) * f


    def dec(self, inp, corruption_type=None, corruption_level=0, border_mode = 'valid'):

        # Corrupt input
        inp = corrupt(inp, corruption_type, corruption_level)

        # Apply filter and return
        return conv(inp, self.W[:,:,::-1,::-1].transpose(1,0,2,3), border_mode=border_mode) * self.scale


    def cost(self, inp, dec, weights = cast_x(1)):
        return ((cast_x(0.5)*(dec-inp)**2).mean(3).mean(2).mean(1)*weights).mean()


    def learn(self, inp, trainer, inp_corruption_type=None, inp_corruption_level=0, hid_corruption_type=None, hid_corruption_level=0, cost_weight = cast_x(1), learn_scale_first=False, debug_path=None, nb_frames=None):

        if trainer:
            # Build noisy autoencoder for training
            train_enc = self(inp, inp_corruption_type, inp_corruption_level, 'full')
            train_dec = self.dec(train_enc, hid_corruption_type, hid_corruption_level)
            train_cost = self.cost(inp, train_dec, cost_weight)

            # Build noiseless autoencoder for validation
            valid_enc = self(inp, border_mode = 'full')
            valid_dec = self.dec(valid_enc)
            valid_cost = self.cost(inp, valid_dec, cost_weight)

            # Quick training for weight scaling
            if learn_scale_first:
                lookback = trainer.lookback
                momentum = trainer.momentum
                trainer.lookback = int(ceil(trainer.lookback / 20.))
                trainer.momentum = 0
                trainer([self.scale], train_cost, valid_cost, model_id=self.model_id + '_scaling').learn()
                trainer.lookback = lookback
                trainer.momentum = momentum

            debug_args = dd()
            debug_args.debug_path = debug_path
            debug_args.nb_frames = nb_frames
            debug_args.prefix = 'unsupervised'
            self.trainer = trainer(self.params.values(), train_cost, valid_cost, model_id=self.model_id,
                                   additionnal_updates = self.additionnal_update(),
                                   debug_calls=(self.debug_call, debug_args),
                                   debug_nodes = dd({'unsupervised_'+self.model_id+'_encoder_act_trainset':train_enc}))

        # Learn model
        self.trainer.learn()

    def additionnal_update(self):
        return th.function(inputs=[], outputs=[], updates=dd({self.W:self.W/T.sqrt((self.W**2).sum(3).sum(2).sum(1))[:,None,None,None]}))

    def debug_call(self, debug_path=None, nb_frames=None, prefix=''):

        if not debug_path:
            return

        # Output path
        make_dir(debug_path)

        # Convert weight matrix into filter imagelets and output to file
        W = self.W.get_value()
        W = W.reshape((W.shape[0], nb_frames, W.shape[1]/nb_frames, W.shape[2], W.shape[3]))
        W = W.transpose(1, 0, 2, 3, 4)
        for i, w in enumerate(W):
            convfilterstoimg(w[:,:,::-1,::-1], fn=join(debug_path, '{}_layer{}_{}_frame_{}.tif'.format(prefix, self.layer_id, str(self.W), i)))


    def private_attrname(self, name):
        return '_'+self.__class__.__name__+'__'+name
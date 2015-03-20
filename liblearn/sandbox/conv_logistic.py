# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 22:52:17 2013

@author: francis
"""

import os
import theano                       as th
import numpy                        as np
import theano.tensor                as T

from theano.tensor.nnet.conv        import conv2d as conv
from theano.tensor.nnet             import softmax
from theano.printing                import Print as thprint
from learn.utils.corrupt            import corrupt
from utils.path                     import make_dir

class conv_logistic(object):
    def __init__(self, params, inp, labels, inp_image_sz, inp_filter_sz, lab_image_sz, lab_filter_sz, nb_classes, dropout_level=0):

        #        print '        params = {}'.format(params)
        #        print '        inp = {}'.format(inp)
        #        print '        labels = {}'.format(labels)
        #        print '        image_sz = {}'.format(image_sz)
        #        print '        filter_sz = {}'.format(filter_sz)
        #        print '        dropout_level = {}'.format(dropout_level)

        # Model parameters
        self.params = params
        self.inp = inp
        self.labels = labels
        self._inp_image_sz = inp_image_sz
        self._inp_filter_sz = inp_filter_sz
        self._lab_image_sz = lab_image_sz
        self._lab_filter_sz = lab_filter_sz
        self.nb_classes = nb_classes
        self.dropout_level = dropout_level

        #  Make label and model
        self._update_labels()
        self._update_model()

    def _update_labels(self):
        if self._lab_filter_sz[2] == self._lab_image_sz[2] and self._lab_filter_sz[3] == self._lab_image_sz[3]:
            self.conv_labels = self.labels.flatten(2).dimshuffle(0, 1, 'x', 'x')

        elif self._lab_filter_sz[2] < self._lab_image_sz[2] or self._lab_filter_sz[3] < self._lab_image_sz[3]:

            # Make sure input lab_filter_sz is correct
            assert(self._lab_filter_sz[0] == self._lab_filter_sz[2]*self._lab_filter_sz[3])
            assert(self._lab_filter_sz[1] == 1)

            # Build selection array
            lab_filters = np.zeros(self._lab_filter_sz, dtype = th.config.floatX)

            # Set filters
            s = np.arange(self._lab_filter_sz[0])
            lab_filters[s, 0, (s // self._lab_filter_sz[3]) % self._lab_filter_sz[2], s % self._lab_filter_sz[3]] = 1

            # Create shared variable
            lab_filters = th.shared(lab_filters)

            # Modify input labels
            self.conv_labels = conv(self.labels.astype(th.config.floatX), lab_filters, image_shape = self._lab_image_sz, filter_shape = self._lab_filter_sz, border_mode='valid')
            self.conv_labels = self.conv_labels.astype(self.labels.dtype)

        else :
            raise ValueError, 'Unhandeled case'


    def _update_model(self):

        # Corrupt input
        corr_inp = corrupt(self.inp, 'zeromask', self.dropout_level)

        # Apply convolution
        out = conv(corr_inp, self.params.W, border_mode='valid') + self.params.B.dimshuffle('x', 0, 'x', 'x')

        # Remember convolution output shape
        out_shape = out.shape
        #out_shape = thprint("out.shape = ")(out.shape)

        # Reshape to softmax format (2D)
        out = out.dimshuffle(0, 2, 3, 1).reshape((out.size // self.nb_classes, self.nb_classes))

        # Compute class probability
        self.prob = softmax(out)

        # Class prediction
        self.pred = T.argmax(self.prob, axis=1).reshape((out_shape[0], out_shape[2], out_shape[3], out_shape[1]//self.nb_classes)).dimshuffle(0, 3, 1, 2)

        # Compute softmax cost
        self.cost = -T.mean(T.log(self.prob[T.arange(self.prob.shape[0]), self.conv_labels.dimshuffle(0,2,3,1).flatten()]))

        # Reshape class prob to convolutional format
        self.prob = self.prob.reshape((out_shape[0], out_shape[2], out_shape[3], out_shape[1])).dimshuffle(0, 3, 1, 2)

        # Prediction error
        self.error = T.mean(T.neq(self.pred, self.conv_labels))


    @property
    def output_sz(self):

        # Compute output size
        return (self._inp_image_sz[0], self._inp_filter_sz[0], self._inp_image_sz[2]-self._inp_filter_sz[2]+1, self._inp_image_sz[3]-self._inp_filter_sz[3]+1)


    @property
    def inp_image_sz(self):

        # Returns input size
        return self._inp_image_sz


    @inp_image_sz.setter
    def inp_image_sz(self, sz):

        if len(sz) != 4:
            raise ValueError, "len(sz) should be == 4.  Instead it is {} and sz = {}".format(len(sz), sz)

        # Set new input image input size
        self._inp_image_sz = sz

        # Update model
        self._update_model()


    @property
    def lab_image_sz(self):

        # Returns input size
        return self._lab_image_sz


    @lab_image_sz.setter
    def lab_image_sz(self, sz):

        if len(sz) != 4:
            raise ValueError, "len(sz) should be == 4.  Instead it is {} and sz = {}".format(len(sz), sz)

        # Set new label image size
        self._lab_image_sz = sz

        # Update model
        self._update_labels()
        self._update_model()



class conv_logistic_params(object):
    def __init__(self, filter_sz, layer_id=''):
        """
        Initializes gated dae parameters

        Parameters
        ----------
        filter_sz:      4-dimension tuple of integers
                        filter size (nb filters, stack size, nb lines, nb columns)
                        for input layer : stack size == nb image channels
                        nb_filters corresponds to the number of classes
        layer_id        string representable type
                        string identification (useful for debugging)
        """

        # Hyperparameters
        self.filter_sz = filter_sz

        # Parameter list
        self._params = []

        # Logistic weights
        self.W = th.shared(np.random.uniform(-0.01, 0.01, filter_sz).astype(th.config.floatX), name='logistic_weights_' + str(layer_id))
        #self.W = th.shared(np.zeros(filter_sz, dtype = th.config.floatX), name='logistic_weights_' + str(layer_id))
        self._params += [self.W]

        # Logistic biases
        self.B = th.shared(np.zeros(filter_sz[0], dtype = th.config.floatX), borrow=True, name='logistic_biases_'+str(layer_id))
        self._params += [self.B]


    def __call__(self):
        """
        Returns a list of selected parameters
        """
        return self._params

    def export(self, path, which=None):
        """
        Export selected parameters matrices to npy files

        Parameters
        ----------
        which:      string or NoneType
                    if which == encoder, then export only encoder parameters.
                    otherwise, it returns all parameters
        """
        make_dir(path)
        for param in self._params:
            fn = os.path.join(path, "{}.npy".format(str(param)))
            np.save(fn, param.get_value(borrow=True))


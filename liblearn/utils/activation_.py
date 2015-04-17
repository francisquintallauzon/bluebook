# -*- coding: utf-8 -*-
"""
Created on Thu May 16 13:32:58 2013

@author: root
"""
import theano.tensor    as T


class activationbase(object):
    def __eq__(self, other):
        if isinstance(other, str):
            return str(self) == str(other)
        else :
            return self==other


class _tanh(activationbase):

    def __call__(self, inp):
        return T.tanh(inp)

    def __repr__(self):
        return 'tanh'


class _relu(activationbase):
    def __call__(self, inp):
        return T.maximum(T.cast(0, dtype=inp.dtype), inp)

    def __repr__(self):
        return 'relu'


class _linear(activationbase):

    def __call__(self, inp):
        return inp

    def __repr__(self):
        return 'linear'


class _step(activationbase):
        
    def __call__(self, inp):
        return T.cast(inp > T.cast(0, dtype=inp.dtype) , inp.dtype)

    def __repr__(self):
        return 'step'


class _sigmoid(activationbase):

    def __call__(self, inp):
        return T.nnet.sigmoid(inp)

    def __repr__(self):
        return 'sigmoid'


class _softplus(activationbase):

    def __call__(self, inp):
        return T.nnet.softplus(inp)

    def __repr__(self):
        return 'softplus'


tanh = _tanh()
relu = _relu()
linear = _linear()
step = _step()
sigmoid = _sigmoid()
softplus = _softplus()


def activation(activation_type):
    """
    Initialize an activation object.

    Parameters
    ----------
    activation_type:    str, ['tanh' or 'rectified linear']
                        specifies the activation object's type
    """

    for act in [tanh, relu, linear, step, sigmoid, softplus]:
        if activation_type == act:
            return act

    # Activation_type not found in current list
    raise NotImplementedError("{} is an unrecognized activation type".format(activation_type))

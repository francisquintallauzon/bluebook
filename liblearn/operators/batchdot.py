# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 20:53:56 2014

@author: francis
"""


from theano        import scan
from theano.tensor import dot



def batchdot(x, y):

    if x.ndim == 2 and y.ndim == 3:
        result, _ = scan(fn=lambda b, a: dot(a, b), outputs_info=None, sequences=[y], non_sequences=[x])
    elif x.ndim == 3 and y.ndim == 2:
        result, _ = scan(fn=lambda a, b: dot(a, b), outputs_info=None, sequences=[x], non_sequences=[y])
    elif x.ndim == 3 and y.ndim == 3:
        result, _ = scan(fn=lambda a, b: dot(a, b), outputs_info=None, sequences=[x,y], non_sequences=None)
    else :
        raise NotImplementedError("x.ndim (={}) and y.ndim (={}) must be 2 or 3".format(x.ndim, y.ndim))

    return result


if __name__ == '__main__':
    import numpy as np
    import theano as th
    import theano.tensor as T

    x = T.matrix('x', dtype=th.config.floatX)
    y = T.tensor3('y', dtype=th.config.floatX)

    fn = th.function(inputs = [x,y], outputs = batchdot(x,y))

    x = np.random.uniform(0, 1, (3,4)).astype(th.config.floatX)
    y = np.random.uniform(0, 1, (2,4,3)).astype(th.config.floatX)

    trad = np.empty((2,3,3), dtype = th.config.floatX)
    for i in range (y.shape[0]):
        trad[i] = np.dot(x, y[i])

    thea = fn(x,y)
    
    print(thea.shape)

    print(trad - thea)


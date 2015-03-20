# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 22:43:26 2013

@author: francis
"""

import numpy             as np
import theano            as th
import theano.tensor     as T
from os.path             import join
from learn.utils         import shared_x
from learn.utils         import cast_x
from learn.layer         import base
from utils.path          import make_dir
from learn.utils         import filterstoimg
from utils.function      import propertydescriptor

class canonical(base):
    data = propertydescriptor()
    def __init__(self, nb_inp, layer_id='', do_dc=True, do_std=True, nb_pca=None, pov=None, whiten=True, dc=None, std=None, pca=None):
        base.__init__(self, layer_id, dc=dc, std=std, pca=pca)

        # Security checks
        if nb_pca > 0 and pov > 0 :
            raise ValueError, "Both self.params.nb_pca (={}) and pov (={}) are valid.  Exactly one must have a valid value".format(nb_pca, pov)

        self.nb_inp = nb_inp
        self.nb_pca = nb_pca
        self.pov = pov
        self.whiten = whiten
        self.do_dc = do_dc
        self.do_std = do_std
        self.do_pca = self.nb_pca > 0 or self.pov > 0 or self.pca

        if not self.dc and do_dc:
            self.dc = shared_x(np.zeros(nb_inp), name = 'dc')

        if not self.std and do_std:
            self.std = shared_x(np.zeros(nb_inp), name = 'std')

        if self.do_pca and not self.pca:
            self.pca = th.shared(np.zeros((nb_inp, nb_pca if self.nb_pca else 0), dtype=th.config.floatX), name = 'pca')

    def __call__(self, inp):

        if self.do_dc:
            inp -= self.dc

        if self.do_std:
            inp /= self.std

        if self.do_pca:
            inp = T.dot(inp, self.pca)

        return inp


    def learn(self, model_inp, layer_inp, data, nb_iterations=1):

        self.data = data

        print '    Learning {}'.format(self.__class__.__name__)

        if self.do_dc:
            # DC centering
            updates = [(self.dc, self.dc + layer_inp.mean(0) / cast_x(nb_iterations))]
            fn = th.function(inputs=[model_inp], updates = updates)
            for i in range(nb_iterations):
                fn(self.data)
            print '      - dc centering:  mean(dc) = {:0.2f};  std(dc) = {:0.2f}'.format(self.dc.get_value().mean(), self.dc.get_value().std())

        if self.do_std:
            # Global contrast normalization
            updates = [(self.std, self.std + (layer_inp**2).mean(0))]
            fn = th.function(inputs=[model_inp], updates = updates)
            for i in range(nb_iterations):
                fn(self.data)
            th.function(inputs=[], updates = [(self.std, T.sqrt(self.std / cast_x(nb_iterations) - self.dc**2))])()
            th.function(inputs=[], updates = [(self.std, self.std + self.std.mean()/cast_x(1000))])()
            print '      - contrast nrm:  mean(std) = {:0.2f};  std(std) = {:0.2f}'.format(self.std.get_value().mean(), self.std.get_value().std())


        # Learn PCA only if the number of components or percentage of variance is positive
        if self.do_pca:

            cov = shared_x(np.zeros((self.nb_inp, self.nb_inp)), name = 'cov_layer')

            mat = (layer_inp - self.dc) if self.do_dc else layer_inp
            mat = mat / self.std if self.do_std else mat
            updates = [(cov, cov + T.dot(mat.T, mat) / cast_x(nb_iterations * mat.shape[0]))]
            fn = th.function(inputs=[model_inp], updates=updates)
            for i in range(nb_iterations):
                fn(self.data)

            l, w = np.linalg.eig(cov.get_value())

            # In case of complex values
            l = abs(l)

            # Sort eigenvalues
            sort = np.argsort(l)[::-1]

            if self.nb_pca > 0:
                # Keep specified number of pca components
                sort = sort[:self.nb_pca]
            else :
                # Keep the number of pca components necessary to retain the specified percentage of variance
                csum = l[sort].cumsum()
                csum = csum / csum[-1]
                sort = sort[:(np.where(csum >= self.pov)[0][0]+1)]
                self.nb_pca = len(sort)

            print '      - pca with {} components retains {:.3f}% of variance'.format(len(sort), l[sort].sum() / l.sum()*100)

            # Extract most important eigen vectors and whiten if requested by the user
            if self.whiten:
                w  = w[:, sort] / np.sqrt(l[sort])
            else :
                w = w[:, sort]

            self.pca.set_value(w.astype(th.config.floatX))

    def debug_call(self, debug_path=None, patch_sz=None, prefix=''):

        class functor(object):
            def __init__(self, obj, debug_path=None, patch_sz=None, prefix=''):
                self.W = obj.pca
                self.layer_id = obj.layer_id
                self.debug_path = debug_path
                self.patch_sz = patch_sz
                self.prefix = prefix

                # Make output directory
                make_dir(self.debug_path)

            def __call__(self):
                if self.W is None :
                    return

                W = self.W.get_value()
                nb_frames = W.shape[0] / np.prod(patch_sz)
                W = W.transpose(1,0)
                W = W.reshape((W.shape[0], nb_frames, -1))
                W = W.transpose(1, 2, 0)
                for i, w in enumerate(W):
                    filterstoimg(w, patch_sz, fn=join(self.debug_path, '{}{}_encoder_weights_frame_{}.tif'.format(prefix, self.layer_id, i)))

        return None if debug_path is None else functor(self, debug_path, patch_sz, prefix)



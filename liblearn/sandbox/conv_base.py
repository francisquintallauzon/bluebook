# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 17:29:00 2014

@author: francis
"""
import theano                       as th
import numpy                        as np
from os.path                        import join
from utils.path                     import make_dir
from learn.utils.display            import convfilterstoimg

class conv_base(object):

    def __getattr__(self, key):
        return self.params[key]

    def __setattr__(self, key, val):
        self.params[key] = val

    def output_sz(self, input_sz):
        return (input_sz[0], self.filter_sz[0], input_sz[2]-self.filter_sz[2]+1, input_sz[3]-self.filter_sz[3]+1)

    def export(self, export_path):
        """
        Export optimizable parameters to the specified export path
        """
        make_dir(export_path)
        for key, param in self.__params.items():
            np.save(join(export_path, "{}.npy".format(key)), param.get_value())

    def fromfile(self, import_path):
        """
        Import tunable parameters from file
        """
        for key, param in self.__params.items():
            th.shared(np.load(join(import_path, "{}.npy".format(key))), name=['biases'+self.layer_id])

    def set_debug_info(self, debug_path, pca_weights=None, prefix=None):
        make_dir(debug_path)
        self.__debug = True
        self.__debug_path = debug_path
        self.__prefix = '' if prefix == None else prefix  + '_'

    def debug_call(self):
        if self.__debug:
            W = self.__params.weights.get_value()
            fn = join(self.__debug_path, self.__prefix + str(self.W) + '.tif')
            convfilterstoimg(W, fn)

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 17:27:05 2015
@author: francis
"""

import numpy                   as np

from liblearn.layer            import conv_maxpool
from liblearn.layer            import conv_preprocess
from liblearn.layer            import conv_vanilla
from liblearn.layer            import hidden
from liblearn.layer            import logistic
from liblearn.layer            import corrupt
from liblearn.layer            import conv_batchnorm

from libutils.dict             import dd
from libutils.path             import make_dir
from os.path                   import join



class model(object):

    def __init__(self, hp, input_sz, import_path=None):
        self.hp = hp
        self.input_sz = input_sz

        self.layers = []

        if import_path :
            print("\nBuilding learning model from {}".format(import_path))
        else:
            print("\nBuilding learning model")

        for i, lhp in list(self.hp.layers.items()):

            path = join(import_path, '{:03d}_{}'.format(i, lhp.type)) if import_path else None

            if lhp.type == "conv_preprocess":
                if path:
                    layer = conv_preprocess.load(path)
                else:
                    layer = conv_preprocess(lhp.nb_channels, lhp.nb_pretrain_iterations)

            elif lhp.type == "conv_batchnorm":
                if path:
                    layer = conv_batchnorm.load(path)
                else:
                    layer = conv_batchnorm(input_sz[1:])

            elif lhp.type == "conv_vanilla":
                if path:
                    layer = conv_vanilla.load(path)
                else:
                    layer = conv_vanilla(lhp.activation, (lhp.nb_filters, input_sz[1]) + lhp.filter_sz)

            elif lhp.type == "conv_maxpool":
                if path:
                    layer = conv_maxpool.load(path)
                else:
                    layer = conv_maxpool(lhp.downsample_sz)

            elif lhp.type == "corrupt":
                if path:
                    layer = corrupt.load(path)
                else:
                    layer = corrupt(lhp.corruption_type, lhp.corruption_level)

            elif lhp.type == "hidden":
                if path:
                    layer = hidden.load(path)
                else:
                    layer = hidden(lhp.activation, np.prod(input_sz[1:]), lhp.nb_hid)

            elif lhp.type == "logistic":
                if path:
                    layer = logistic.load(path)
                else:
                    layer = logistic(np.prod(input_sz[1:]), lhp.nb_out)

            else:
                raise ValueError("'{}' : unrecognized layer type".format(lhp.type))

            print("    layer {:2d} : {} --> {} on {}".format(i, input_sz[1:], layer.shape(input_sz)[1:], layer))

            input_sz = layer.shape(input_sz)

            self.layers += [layer]

    @property
    def name(self):
        return self.hp.name


    def __call__(self, inp, mode):
        for layer in self.layers:
            inp = layer(inp, mode)
        return inp


    def pretrain(self, data_input, model_input, train_data):
        print('\nPretraining model')
        for i, layer in enumerate(self.layers):
            if layer.name == 'conv_preprocess':
                layer.learn(data_input, model_input, train_data)
            model_input = layer(model_input, mode='train')


    def posttrain(self, data_input, model_input, train_data):
        print('\nPosttraining model')
        for layer in self.layers:
            if layer.name == 'conv_batchnorm':
                layer.learn(data_input, model_input, train_data)
            model_input = layer(model_input, mode='train')


    def shape(self, input_sz):
        for layer in self.layers:
            input_sz = layer.shape(input_sz)
        return input_sz


    @property
    def params(self):
        params = []
        for layer in self.layers:
            params += layer.params
        return params

    @property
    def optimizables(self):
        params = []
        for layer in self.layers:
            params += layer.optimizables
        return params


    def export(self, path):

        make_dir(path)

        hparams = dd({'hp':self.hp,
                      'input_sz':self.input_sz})

        hparams.dump(join(path, 'hparams.pkl'))

        for i, layer in enumerate(self.layers):
            layer.export(join(path, '{:03d}_{}'.format(i, layer.name)))


    @classmethod
    def load(cls, path):
        return cls(path, **dd.load(join(path, 'hparams.pkl')))



#    def debug_call(self, debug_path=None, patch_sz=None, pca=None, prefix=''):
#
#        class functor(object):
#            def __init__(self, obj, debug_path=None, patch_sz=None, pca=None, prefix=''):
#                self.W = obj.W
#                self.layer_id = obj.layer_id
#                self.debug_path = join(debug_path, 'filters')
#                self.patch_sz = patch_sz
#                self.pca = pca
#                self.prefix = prefix
#
#                # Make output directory
#                make_dir(self.debug_path)
#
#            def __call__(self):
#                W = T.dot(pca, self.W).eval() if pca else self.W.get_value()
#                nb_frames = W.shape[0] / np.prod(patch_sz)
#                W = W.transpose(1,0)
#                W = W.reshape((W.shape[0], nb_frames, -1))
#                W = W.transpose(1, 2, 0)
#                for i, w in enumerate(W):
#                    filterstoimg(w, patch_sz, fn=join(self.debug_path, '{}{}_encoder_weights_frame_{}.tif'.format(prefix, self.layer_id, i)))
#
#        return None if debug_path is None else functor(self, debug_path, patch_sz, pca, prefix)

if __name__ == '__main__':

    import results
    #path = join(results.path(), 'catsanddogs', 'exp000', '20150403_16h13m02s_452149', '00000', 'hp.pkl')
    path = join(results.path(), 'catsanddogs', 'exp001', '20150407_17h46m23s_803000', '00074', 'hp.pkl')
    hp = dd.load(path)
    m = model(hp.model, (100, 3, 90, 90))


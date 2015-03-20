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

from libutils.dict             import dd
from libutils.path             import make_dir
from os.path                   import join



class model(object):

    def __init__(self, hp, input_sz, import_path=None):
        self.hp = hp
        self.input_sz = input_sz

        self.layers = dd()
        self.layers.train = []
        self.layers.test = []
        self.layers.all = []

        if import_path :
            print "\nBuilding learning model from {}".format(import_path)  
        else:
            print "\nBuilding learning model"

        for i, lhp in self.hp.layers.items():

            path = join(import_path, '{:03d}_{}'.format(i, lhp.type)) if import_path else None
            
            if lhp.type == "conv_preprocess":
                if path:
                    layer = conv_preprocess.load(path)
                else:
                    layer = conv_preprocess(lhp.nb_channels, lhp.nb_pretrain_iterations)

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
                raise ValueError, "{} un recognized layer type".format(lhp.type)

            print "    layer {:2d} : {} --> {} on {}".format(i, input_sz[1:], layer.shape(input_sz)[1:], layer)

            input_sz = layer.shape(input_sz)

            if 'mode' in lhp:
                if 'train' in lhp.mode:
                    self.layers.train += [layer]
                if 'test' in lhp.mode:
                    self.layers.test += [layer]
            else :
                self.layers.train += [layer]
                self.layers.test += [layer]
            self.layers.all += [layer]


    @property
    def name(self):
        return self.hp.name


    def __call__(self, inp, mode):
        for layer in self.layers[mode]:
            inp = layer(inp)
        return inp


    def pretrain(self, data_input, model_input, train_data):
        print '\nPretraining model'
        for layer in self.layers.train:
            if layer.type == 'conv_preprocess':
                layer.learn(data_input, model_input, train_data)
            model_input = layer(model_input)


    def shape(self, input_sz):
        for layer in self.layers.test:
            input_sz = layer.shape(input_sz)
        return input_sz


    def get_params(self, trainable_only=False):
        params = []
        for layer in self.layers.train:
            if trainable_only:
                if layer.type == 'conv_preprocess':
                    continue
            params += layer.params.values()
        return params


    def export(self, path):
        
        make_dir(path)

        hparams = dd({'hp':self.hp,
                      'input_sz':self.input_sz})
                      
        hparams.dump(join(path, 'hparams.pkl'))
        
        for i, layer in enumerate(self.layers.all):
            layer.export(join(path, '{:03d}_{}'.format(i, layer.name)))

    @classmethod
    def load(cls, path):
        return cls(path, **dd.load(join(path, 'hparams.pkl')))






if __name__ == '__main__':
    pass


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


    # Feature learning layers
    hp = dd()
    hp.name = 'convnet'
    hp.layers = dd()

    # Preprocess layer
    i = 0
    hp.layers[i] = dd()
    hp.layers[i].type = 'conv_preprocess'
    hp.layers[i].nb_channels = 3
    hp.layers[i].nb_pretrain_iterations = 1 #int(1. / hp.features.load_ratio) // 5

    # Convolutional layer
    i += 1
    hp.layers[i] = dd()
    hp.layers[i].type = 'conv_vanilla'
    hp.layers[i].activation = "relu"
    hp.layers[i].nb_filters = 16
    hp.layers[i].filter_sz =  (3, 3)

    # Max pooling layer
    i += 1
    hp.layers[i] = dd()
    hp.layers[i].type = 'conv_maxpool'
    hp.layers[i].downsample_sz = 2


    # Convolutional layer
    i += 1
    hp.layers[i] = dd()
    hp.layers[i].type = 'conv_vanilla'
    hp.layers[i].activation = "relu"
    hp.layers[i].nb_filters = 32
    hp.layers[i].filter_sz =  (3, 3)

    # Convolutional layer
    i += 1
    hp.layers[i] = dd()
    hp.layers[i].type = 'conv_vanilla'
    hp.layers[i].activation = "relu"
    hp.layers[i].nb_filters = 32
    hp.layers[i].filter_sz =  (3, 3)

    # Max pooling layer
    i += 1
    hp.layers[i] = dd()
    hp.layers[i].type = 'conv_maxpool'
    hp.layers[i].downsample_sz = 2

    # Convolutional layer
    i += 1
    hp.layers[i] = dd()
    hp.layers[i].type = 'conv_vanilla'
    hp.layers[i].activation = "relu"
    hp.layers[i].nb_filters = 64
    hp.layers[i].filter_sz =  (5, 5)

    # Max pooling layer
    i += 1
    hp.layers[i] = dd()
    hp.layers[i].type = 'conv_maxpool'
    hp.layers[i].downsample_sz = 2

    # Convolutional layer
    i += 1
    hp.layers[i] = dd()
    hp.layers[i].type = 'conv_vanilla'
    hp.layers[i].activation = "relu"
    hp.layers[i].nb_filters = 64
    hp.layers[i].filter_sz =  (5, 5)

    # Fully connected layer
    i += 1
    hp.layers[i] = dd()
    hp.layers[i].type = 'hidden'
    hp.layers[i].activation = "relu"
    hp.layers[i].nb_hid = 256 # int(10**np.random.uniform(log(128)/log(10), log(512)/log(10)))

    # Logistic layer
    i += 1
    hp.layers[i] = dd()
    hp.layers[i].type = 'logistic'
    hp.layers[i].nb_out = 2


    m = model(hp, (1000, 3, 100, 100))
    m.export('./modeltest')
    m = model(hp, (1000, 3, 100, 100), './modeltest')


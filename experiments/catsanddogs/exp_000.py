# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 11:17:49 2015
@author: francis
"""


import theano as th
import theano.tensor as T
import numpy as np
import datasets
from math import log, exp
from datetime import datetime
from os.path import join
from source_ import source
from features_ import features
from liblearn import model
from liblearn.train import sgd
from libutils.dict import dd
from liblearn.loss import cross_entropy, error
from liblearn.utils import float_x
from libutils.debug import lineno


class experiment(object):
    def __init__(self, path=None):

        # Generate hyper parameters
        hp = self.generate_hp(path)
        self.hp = hp
        
    def run(self):
        hp = self.hp        

        # Initialize dataset
        self.source = source(hp.dataset.path)

        # Initialize feature extraction layer
        self.features = features(self.source, **hp.features)

        # Model input and labels
        self.input = T.tensor4("input_tensor4", float_x)
        self.label = T.ivector("label_vector")

        # Create trainer object
        self.trainer = sgd(self.input, self.features.train, self.features.valid,  \
                           log=dd(), output_path=join(hp.meta.path, hp.meta.result_folder), **hp.trainer)

        # Initialize experiment
        self.model = model(hp.model, self.features.shape())

        # Greedy layer-wise pre-training
        self.model.pretrain(self.input, self.input, self.features.train)

        # Class probability output
        train_prob = self.model(self.input, 'train')
        valid_prob = self.model(self.input, 'test')

        # Loss functions
        train_loss = cross_entropy(train_prob, self.label)
        valid_loss = cross_entropy(valid_prob, self.label)

        # Error functions
        train_error = error(train_prob, self.label)
        valid_error = error(valid_prob, self.label)

        # Model wide training
        self.trainer(self.model.get_params(True), train_loss, valid_loss, self.label, train_error, valid_error, 'exp_000').learn()

        
        # Export parameters
        # self.model.export(join(hp.meta.path, hp.meta.export_folder))


    def generate_hp(self, path):

        # Seed random number generator
        np.random.seed(datetime.now().microsecond)

        # Hyperparameters
        hp = dd()

        # Set debug mode
        hp.debug = False

        # Experiment parameters
        hp.meta = dd()
        hp.meta.path = path 
        hp.meta.result_folder = './'
        hp.meta.export_folder = 'export'

        # Dataset parameters
        hp.dataset = dd()
        hp.dataset.path = join(datasets.path(), 'catsanddogs')

        # Feature extraction parameters
        hp.features = dd()
        hp.features.load_ratio = 0.05
        hp.features.patch_sz = (84,84,3)
        hp.features.reshape_sz = (96,96,3)
        
        # Feature learning layers
        hp.model = dd()
        
        # Preprocess layer
        i = 1
        hp.model[i] = dd()
        hp.model[i].type = 'conv_preprocess'
        hp.model[i].nb_channels = 3
        hp.model[i].nb_pretrain_iterations = 1 #int(1. / hp.features.load_ratio) // 5
        
#        # Dropout layer
#        i += 1
#        hp.model[i] = dd()
#        hp.model[i].type = 'corrupt'
#        hp.model[i].mode = 'train'
#        hp.model[i].corruption_type = 'zeromask'
#        hp.model[i].corruption_level = 0.5
        
        # Convolutional layer
        i += 1
        hp.model[i] = dd()
        hp.model[i].type = 'conv_vanilla'
        hp.model[i].activation = "relu"
        hp.model[i].nb_filters = 16
        hp.model[i].filter_sz =  (3, 3)

        # Max pooling layer
        i += 1
        hp.model[i] = dd()
        hp.model[i].type = 'conv_maxpool'
        hp.model[i].downsample_sz = 2

#        # Dropout layer
#        i += 1
#        hp.model[i] = dd()
#        hp.model[i].type = 'corrupt'
#        hp.model[i].mode = 'train'
#        hp.model[i].corruption_type = 'zeromask'
#        hp.model[i].corruption_level = 0.5

        # Convolutional layer
        i += 1
        hp.model[i] = dd()
        hp.model[i].type = 'conv_vanilla'
        hp.model[i].activation = "relu"
        hp.model[i].nb_filters = 32
        hp.model[i].filter_sz =  (3, 3)
        

#        # Dropout layer
#        i += 1
#        hp.model[i] = dd()
#        hp.model[i].type = 'corrupt'
#        hp.model[i].mode = 'train'
#        hp.model[i].corruption_type = 'zeromask'
#        hp.model[i].corruption_level = 0.5

        # Convolutional layer
        i += 1
        hp.model[i] = dd()
        hp.model[i].type = 'conv_vanilla'
        hp.model[i].activation = "relu"
        hp.model[i].nb_filters = 32
        hp.model[i].filter_sz =  (3, 3)

        # Max pooling layer
        i += 1
        hp.model[i] = dd()
        hp.model[i].type = 'conv_maxpool'
        hp.model[i].downsample_sz = 2               
        
#        # Dropout layer
#        i += 1
#        hp.model[i] = dd()
#        hp.model[i].type = 'corrupt'
#        hp.model[i].mode = 'train'
#        hp.model[i].corruption_type = 'zeromask'
#        hp.model[i].corruption_level = 0.5

        # Convolutional layer
        i += 1
        hp.model[i] = dd()
        hp.model[i].type = 'conv_vanilla'
        hp.model[i].activation = "relu"
        hp.model[i].nb_filters = 64
        hp.model[i].filter_sz =  (5, 5)
        
        # Max pooling layer
        i += 1
        hp.model[i] = dd()
        hp.model[i].type = 'conv_maxpool'
        hp.model[i].downsample_sz = 2                

#        # Dropout layer
#        i += 1
#        hp.model[i] = dd()
#        hp.model[i].type = 'corrupt'
#        hp.model[i].mode = 'train'
#        hp.model[i].corruption_type = 'zeromask'
#        hp.model[i].corruption_level = 0.5

        # Convolutional layer
        i += 1
        hp.model[i] = dd()
        hp.model[i].type = 'conv_vanilla'
        hp.model[i].activation = "relu"
        hp.model[i].nb_filters = 64
        hp.model[i].filter_sz =  (5, 5)

#        # Dropout layer
#        i += 1
#        hp.model[i] = dd()
#        hp.model[i].type = 'corrupt'
#        hp.model[i].mode = 'train'
#        hp.model[i].corruption_type = 'zeromask'
#        hp.model[i].corruption_level = 0.5


        # Fully connected layer
        i += 1
        hp.model[i] = dd()
        hp.model[i].type = 'hidden'
        hp.model[i].activation = "relu"
        hp.model[i].nb_hid = 256 # int(10**np.random.uniform(log(128)/log(10), log(512)/log(10)))
        
#        # Dropout layer
#        i += 1
#        hp.model[i] = dd()
#        hp.model[i].type = 'corrupt'
#        hp.model[i].mode = 'train'
#        hp.model[i].corruption_type = 'zeromask'
#        hp.model[i].corruption_level = 0.5
       

        # Logistic layer
        i += 1
        hp.model[i] = dd()
        hp.model[i].type = 'logistic'
        hp.model[i].nb_out = 2
        
        # Trainer
        hp.trainer = dd()
        hp.trainer.max_epoch=10000
        hp.trainer.lookback=100
        hp.trainer.minibatch_sz=100
        hp.trainer.init_lr = None
        hp.trainer.incr_lr = None
        hp.trainer.lr = 0.01 #10**np.random.uniform(log(0.001)/log(10), log(0.1)/log(10))
        hp.trainer.log_decay_rate=0
        hp.trainer.momentum = 0.9 #np.random.uniform(0, 0.99)
        hp.trainer.momentum_reset_prob = 0

        print "Save hyperparameters to file"
        hp.dump(join(hp.meta.path, 'hp.pkl'), save_pretty_textfile=True)

        print hp

        return hp

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Runs experiment")
    parser.add_argument('--path', required=True, type=str, help='Experiment result path')
    args = parser.parse_args()

    e = experiment(args.path)
    
    try :
        e.run()
    except (SystemExit, KeyboardInterrupt) as e:
        print '{} in {}...  Exiting gracefully.'.format(e.message, __name__)
        
        
        
    print "Experiment done!"
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 11:17:49 2015
@author: francis
"""

import theano as th
import theano.tensor as T
import numpy as np
from numpy.random import uniform, randint
import datasets
from math import log, exp
from datetime import datetime
from os.path import join
from .source_ import source
from liblearn.iterators  import randompatchiterator
from liblearn import model
from liblearn.train import sgd
from libutils.dict import dd
from liblearn.loss import cross_entropy, error
from liblearn.utils import float_x


class experiment(object):
    def __init__(self, path=None):

        # Generate hyper parameters
        hp = self.generate_hp(path)
        self.hp = hp

    def run(self):
        hp = self.hp

        # Initialize log file
        self.log = dd()

        # Initialize dataset
        self.source = source(hp.dataset.path)

        # Initialize feature extraction layer
        self.trainiterator = randompatchiterator(self.source.splits.train, load_ratio=0.05, dataset_per_epoch=1, corrupt=True, nb_workers=5, **hp.iterator)
        self.validiterator = randompatchiterator(self.source.splits.valid, load_ratio=0.25, dataset_per_epoch=1, corrupt=False, nb_workers=4, **hp.iterator)
        self.testiterator = randompatchiterator(self.source.splits.test, load_ratio=0.25, corrupt=False, nb_workers=1, **hp.iterator)

        # Model input and labels
        self.input = T.tensor4("input_tensor4", float_x)
        self.label = T.ivector("label_vector")

        # Create trainer object
        self.trainer = sgd(self.input, self.trainiterator, self.validiterator,  \
                           log=self.log, output_path=join(hp.meta.path, hp.meta.result_folder), **hp.trainer)

        # Initialize experiment
        self.model = model(hp.model, self.trainiterator.shape)

        # Greedy layer-wise pre-training
        self.model.pretrain(self.input, self.input, self.trainiterator)

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
        self.trainer(self.model.optimizables, train_loss, valid_loss, self.label, train_error, valid_error, self.model.name).learn()

        # Compute final valid and test result
        self.performance_fn = th.function(inputs = [self.input, self.label], outputs = [valid_loss, valid_error], allow_input_downcast=True)

        # Test performance on validation set
        valid_loss, valid_error = self.test(self.validiterator)
        test_loss, test_error = self.test(self.testiterator)

        # Final results to log file
        self.log[self.model.name].cost.final_valid = valid_loss
        self.log[self.model.name].cost.final_test = test_loss
        self.log[self.model.name].error.final_valid = valid_error
        self.log[self.model.name].error.final_test = test_error
        self.log.dump(join(hp.meta.path, hp.meta.result_folder, "out.pkl"), True)

        # Export parameters
        # self.model.export(join(hp.meta.path, hp.meta.export_folder))


    def test(self, iterator):
        total_loss = 0.
        total_error = 0.
        nb_samples = 0
        for (data, labels) in iterator:
            l, e = self.performance_fn(data, labels)
            total_loss += l * data.shape[0]
            total_error += e * data.shape[0]
            nb_samples += data.shape[0]
        return total_loss / nb_samples, total_error / nb_samples


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
        hp.iterator = dd()
        hp.iterator.patch_sz = (90,90,3)
        hp.iterator.reshape_sz = (100,100,3)

        # Feature learning layers
        hp.model = dd()
        hp.model.name = 'convnet'
        hp.model.corruption_type = [None, 'zeromask'][randint(0, 2)]
        hp.model.corruption_level = uniform(0, 0.5)
        hp.model.layers = dd()

        # Preprocess layer
        i = 1
        hp.model.layers[i] = dd()
        hp.model.layers[i].type = 'conv_preprocess'
        hp.model.layers[i].nb_channels = 3
        hp.model.layers[i].nb_pretrain_iterations = 5 

        # Dropout layer
        i += 1
        hp.model.layers[i] = dd()
        hp.model.layers[i].type = 'corrupt'
        hp.model.layers[i].corruption_type = hp.model.corruption_type
        hp.model.layers[i].corruption_level = 0.1

        # Convolutional layer
        i += 1
        hp.model.layers[i] = dd()
        hp.model.layers[i].type = 'conv_vanilla'
        hp.model.layers[i].activation = "relu"
        hp.model.layers[i].nb_filters = 18
        hp.model.layers[i].filter_sz =  (3, 3)

        # Max pooling layer
        i += 1
        hp.model.layers[i] = dd()
        hp.model.layers[i].type = 'conv_maxpool'
        hp.model.layers[i].downsample_sz = 2

        # Dropout layer
        i += 1
        hp.model.layers[i] = dd()
        hp.model.layers[i].type = 'corrupt'
        hp.model.layers[i].corruption_type = hp.model.corruption_type
        hp.model.layers[i].corruption_level = hp.model.corruption_level

        # Convolutional layer
        i += 1
        hp.model.layers[i] = dd()
        hp.model.layers[i].type = 'conv_vanilla'
        hp.model.layers[i].activation = "relu"
        hp.model.layers[i].nb_filters = 36
        hp.model.layers[i].filter_sz =  (3, 3)

        # Dropout layer
        i += 1
        hp.model.layers[i] = dd()
        hp.model.layers[i].type = 'corrupt'
        hp.model.layers[i].corruption_type = hp.model.corruption_type
        hp.model.layers[i].corruption_level = hp.model.corruption_level

        # Convolutional layer
        i += 1
        hp.model.layers[i] = dd()
        hp.model.layers[i].type = 'conv_vanilla'
        hp.model.layers[i].activation = "relu"
        hp.model.layers[i].nb_filters = 36
        hp.model.layers[i].filter_sz =  (3, 3)

        # Max pooling layer
        i += 1
        hp.model.layers[i] = dd()
        hp.model.layers[i].type = 'conv_maxpool'
        hp.model.layers[i].downsample_sz = 2

        # Dropout layer
        i += 1
        hp.model.layers[i] = dd()
        hp.model.layers[i].type = 'corrupt'
        hp.model.layers[i].corruption_type = hp.model.corruption_type
        hp.model.layers[i].corruption_level = hp.model.corruption_level

        # Convolutional layer
        i += 1
        hp.model.layers[i] = dd()
        hp.model.layers[i].type = 'conv_vanilla'
        hp.model.layers[i].activation = "relu"
        hp.model.layers[i].nb_filters = 72
        hp.model.layers[i].filter_sz =  (5, 5)

        # Max pooling layer
        i += 1
        hp.model.layers[i] = dd()
        hp.model.layers[i].type = 'conv_maxpool'
        hp.model.layers[i].downsample_sz = 2

        # Dropout layer
        i += 1
        hp.model.layers[i] = dd()
        hp.model.layers[i].type = 'corrupt'
        hp.model.layers[i].corruption_type = hp.model.corruption_type
        hp.model.layers[i].corruption_level = hp.model.corruption_level

        # Convolutional layer
        i += 1
        hp.model.layers[i] = dd()
        hp.model.layers[i].type = 'conv_vanilla'
        hp.model.layers[i].activation = "relu"
        hp.model.layers[i].nb_filters = 72
        hp.model.layers[i].filter_sz =  (5, 5)

        # Dropout layer
        i += 1
        hp.model.layers[i] = dd()
        hp.model.layers[i].type = 'corrupt'
        hp.model.layers[i].corruption_type = hp.model.corruption_type
        hp.model.layers[i].corruption_level = hp.model.corruption_level

        # Fully connected layer
        i += 1
        hp.model.layers[i] = dd()
        hp.model.layers[i].type = 'hidden'
        hp.model.layers[i].activation = "relu"
        hp.model.layers[i].nb_hid = 256 # int(10**np.random.uniform(log(128)/log(10), log(512)/log(10)))

        # Logistic layer
        i += 1
        hp.model.layers[i] = dd()
        hp.model.layers[i].type = 'logistic'
        hp.model.layers[i].nb_out = 2

        # Trainer
        hp.trainer = dd()
        hp.trainer.max_epoch=10000
        hp.trainer.lookback=20
        hp.trainer.minibatch_sz=100
        hp.trainer.init_lr = None
        hp.trainer.incr_lr = None
        hp.trainer.lr = 0.01 #10**np.random.uniform(log(0.001)/log(10), log(0.1)/log(10))
        hp.trainer.decay_rate=0.985
        hp.trainer.momentum = 0.9 #np.random.uniform(0, 0.99)
        hp.trainer.momentum_reset_prob = 0

        print("Save hyperparameters to file")
        hp.dump(join(hp.meta.path, 'hp.pkl'), save_pretty_textfile=True)

        print(hp)

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
        print('{} in {}...  Exiting gracefully.'.format(e.message, __name__))



    print("Experiment done!")
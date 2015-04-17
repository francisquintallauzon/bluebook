# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 17:02:37 2015

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
from source_ import source
from liblearn.iterators  import randompatchiterator, randompatch
from liblearn import model
from liblearn.train import sgd
from libutils.dict import dd
from liblearn.loss import cross_entropy, error
from liblearn.utils import float_x


class experiment(object):
    def __init__(self, path=None):

        # Generate hyper parameters
        self.hp = self.generate_hp(path)

    def run(self):
        hp = self.hp

        # Initialize log file
        self.log = dd()


        # Model input and labels
        self.input = T.tensor4("input_tensor4", float_x)
        self.label = T.ivector("label_vector")

        # :earing model
        self.model = model(hp.model, self.trainiterator.shape)

        # Class probability output
        train_prob = self.model(self.input, 'train')
        valid_prob = self.model(self.input, 'valid')
        test_prob = self.model(self.input, 'test')

        # Loss functions
        train_loss = cross_entropy(train_prob, self.label)
        valid_loss = cross_entropy(valid_prob, self.label)
        test_loss = cross_entropy(test_prob, self.label)

        # Error functions
        train_error = error(train_prob, self.label)
        valid_error = error(valid_prob, self.label)
        test_error = error(test_prob, self.label)
        
        # Initialize dataset
        self.source = source(hp.dataset.path)        
        
        # Iterators that will read and extract images from the dataset
        self.trainiterator = randompatch(self.source.splits.train, load_ratio=0.05, dataset_per_epoch=1, corrupt=True, nb_workers=1, **hp.iterator)
        self.validiterator = randompatch(self.source.splits.valid, load_ratio=0.1, dataset_per_epoch=1, corrupt=False, nb_workers=1, **hp.iterator)
        self.testiterator = randompatch(self.source.splits.test, load_ratio=0.1, corrupt=False, nb_workers=1, **hp.iterator)        
        
        # Trainer object
        self.trainer = sgd(self.input, self.trainiterator, self.validiterator,  \
                           log=self.log, output_path=join(hp.meta.path, hp.meta.result_folder), **hp.trainer)        

        # Greedy layer-wise pre-training
        self.model.pretrain(self.input, self.input, self.trainiterator)

        # Training
        self.trainer(self.model.optimizables, train_loss, valid_loss, self.label, train_error, valid_error, self.model.name).learn()

        # Greedy layer-wise post-training
        self.model.posttrain(self.input, self.input, self.trainiterator)

        # Compute final valid and test result
        valid_performance_fn = th.function(inputs = [self.input, self.label], outputs = [valid_loss, valid_error], allow_input_downcast=True)
        test_performance_fn = th.function(inputs = [self.input, self.label], outputs = [test_loss, test_error], allow_input_downcast=True)

        # Test performance on validation set
        valid_loss, valid_error = self.test(self.validiterator, valid_performance_fn)
        print("Final valid loss = {:0.4f}, error = {:0.4f}".format(valid_loss, valid_error))

        test_loss, test_error = self.test(self.testiterator, test_performance_fn)
        print("Final test loss = {:0.4f}, error = {:0.4f}".format(test_loss, test_error))

        # Final results to log file
        self.log[self.model.name].cost.final_valid = valid_loss
        self.log[self.model.name].cost.final_test = test_loss
        self.log[self.model.name].error.final_valid = valid_error
        self.log[self.model.name].error.final_test = test_error
        self.log.dump(join(hp.meta.path, hp.meta.result_folder, "out.pkl"), True)

        # Export parameters
        # self.model.export(join(hp.meta.path, hp.meta.export_folder))


    def test(self, iterator, theanofn):
        total_loss = 0.
        total_error = 0.
        nb_samples = 0
        for (data, labels) in iterator:
            l, e = theanofn(data, labels)
            total_loss += l * data.shape[0]
            total_error += e * data.shape[0]
            nb_samples += data.shape[0]

        print('    testing : ')
        print('    total_loss = {}'.format(total_loss))
        print('    total_error = {}'.format(total_error))
        print('    nb_samples = {}'.format(nb_samples))
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
        hp.model.layers = dd()

        # Preprocess layer
        i = 1
        hp.model.layers[i] = dd()
        hp.model.layers[i].type = 'conv_preprocess'
        hp.model.layers[i].nb_channels = 3
        hp.model.layers[i].nb_pretrain_iterations = 1

        # Convolutional layer
        i += 1
        hp.model.layers[i] = dd()
        hp.model.layers[i].type = 'conv_vanilla'
        hp.model.layers[i].activation = "relu"
        hp.model.layers[i].nb_filters = 16
        hp.model.layers[i].filter_sz =  (3, 3)

        # Max pooling layer
        i += 1
        hp.model.layers[i] = dd()
        hp.model.layers[i].type = 'conv_maxpool'
        hp.model.layers[i].downsample_sz = 2

        # Batch normalization layer
        i += 1
        hp.model.layers[i] = dd()
        hp.model.layers[i].type = 'conv_batchnorm'

        # Convolutional layer
        i += 1
        hp.model.layers[i] = dd()
        hp.model.layers[i].type = 'conv_vanilla'
        hp.model.layers[i].activation = "relu"
        hp.model.layers[i].nb_filters = 32
        hp.model.layers[i].filter_sz =  (3, 3)

        # Batch normalization layer
        i += 1
        hp.model.layers[i] = dd()
        hp.model.layers[i].type = 'conv_batchnorm'

        # Convolutional layer
        i += 1
        hp.model.layers[i] = dd()
        hp.model.layers[i].type = 'conv_vanilla'
        hp.model.layers[i].activation = "relu"
        hp.model.layers[i].nb_filters = 32
        hp.model.layers[i].filter_sz =  (3, 3)

        # Max pooling layer
        i += 1
        hp.model.layers[i] = dd()
        hp.model.layers[i].type = 'conv_maxpool'
        hp.model.layers[i].downsample_sz = 2

        # Batch normalization layer
        i += 1
        hp.model.layers[i] = dd()
        hp.model.layers[i].type = 'conv_batchnorm'

        # Convolutional layer
        i += 1
        hp.model.layers[i] = dd()
        hp.model.layers[i].type = 'conv_vanilla'
        hp.model.layers[i].activation = "relu"
        hp.model.layers[i].nb_filters = 64
        hp.model.layers[i].filter_sz =  (5, 5)

        # Max pooling layer
        i += 1
        hp.model.layers[i] = dd()
        hp.model.layers[i].type = 'conv_maxpool'
        hp.model.layers[i].downsample_sz = 2

        # Batch normalization layer
        i += 1
        hp.model.layers[i] = dd()
        hp.model.layers[i].type = 'conv_batchnorm'

        # Convolutional layer
        i += 1
        hp.model.layers[i] = dd()
        hp.model.layers[i].type = 'conv_vanilla'
        hp.model.layers[i].activation = "relu"
        hp.model.layers[i].nb_filters = 64
        hp.model.layers[i].filter_sz =  (5, 5)

        # Batch normalization layer
        i += 1
        hp.model.layers[i] = dd()
        hp.model.layers[i].type = 'conv_batchnorm'

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
        hp.trainer.max_epoch=1000
        hp.trainer.lookback=randint(10, 30)
        hp.trainer.minibatch_sz= 100
        hp.trainer.init_lr = None
        hp.trainer.incr_lr = None
        hp.trainer.lr = 10**np.random.uniform(log(0.01)/log(10), log(0.1)/log(10))
        hp.trainer.decay_rate = uniform(0.985, 1.0)
        hp.trainer.momentum = 10**uniform(log(0.8)/log(10), log(0.99)/log(10))
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
    except  SystemExit:
        print('Exiting following a SystemExit exception')
    except KeyboardInterrupt:
        print('Exiting following a KeyboardInterrupt exception')

    print("Experiment done!")

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 21:10:23 2015

@author: francis
"""
import random
from libutils.dict import dd

class splits(object):

    def __init__(self, examples, valid_ratio=0.1, test_ratio=0.1, shuffle=True):

        # Initialization
        self.__splits = dd([('train', []), ('valid', []), ('test', []), ('full', [])])

        # Extract a copy of the example list
        examples = list(examples)

        # If required, shuffle example list
        if shuffle:
            random.shuffle(examples)

        # Assign examples to splits
        self.__splits.train = examples[:int(len(examples) * (1.-test_ratio-valid_ratio))]
        self.__splits.valid = examples[len(self.__splits.train):len(self.__splits.train)+int(len(examples) * valid_ratio)]
        self.__splits.test  = examples[len(self.__splits.train+self.__splits.valid):]
        self.__splits.full  = examples

        print("\nSplits statistics:")
        for split_id in self.__splits:
            print("        {:7d} items in {} set".format(len(self[split_id]), split_id))

    def __getitem__(self, item):
        return list(self.__splits[item])

    @property
    def train(self):
        return self['train']

    @property
    def valid(self):
        return self['valid']

    @property
    def test(self):
        return self['test']

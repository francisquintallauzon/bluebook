# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 13:23:28 2015
@author: francis
"""

import numpy                   as np
import multiprocessing         as mp
from time                      import time
from math                      import ceil
from scipy.misc                import imresize
from .parallel_random_iterator_ import parallel_random_iterator


class randompatchiterator(object):

    def __init__(self, examples, patch_sz, reshape_sz, corrupt, load_ratio, dataset_per_epoch=1, nb_workers=3):
        self.examples = examples
        self.patch_sz = patch_sz
        self.reshape_sz = reshape_sz
        self.load_ratio = load_ratio
        self.dataset_per_epoch = dataset_per_epoch
        self.corrupt = corrupt
        self.nb_workers = nb_workers

        self.extractor_kwargs = {'patch_sz':patch_sz, 'reshape_sz':reshape_sz, 'corrupt':corrupt}

    def __iter__(self):
        return parallel_random_iterator(self.examples, extractor, self.extractor_kwargs, self.load_ratio, self.dataset_per_epoch, self.nb_workers)

    def __call__(self):
        return self.__iter__()

    @property
    def shape(self):
        nb_examples = int(ceil(self.load_ratio * len(self.examples)))
        return (nb_examples, self.patch_sz[2], self.patch_sz[0], self.patch_sz[1])


class extractor(mp.Process):
    def __init__(self, examples, patch_sz, reshape_sz, corrupt=False):
        mp.Process.__init__(self)
        self.examples = examples
        self.resultqueue = mp.Queue()
        self.patch_sz = patch_sz
        self.reshape_sz = reshape_sz
        self.corrupt = corrupt

        self.readtimequeue = mp.Queue()
        self.distortiontimequeue = mp.Queue()

    def run(self):

        readtime = 0
        distorsiontime = 0

        examples = self.examples

        nb_examples = len(examples)

        images = np.empty((nb_examples,)+self.shape[1:], np.float32)
        labels = np.empty(nb_examples, np.int32)

        for i, example in enumerate(examples):

            # read image from disk
            st = time()
            patch = example.get_image()
            readtime += time()-st

            st = time()

            # resize image
            patch = imresize(patch, self.reshape_sz)

            # Crop patch
            if self.corrupt:
                x = np.random.randint(0, patch.shape[1]-self.patch_sz[1])
                y = np.random.randint(0, patch.shape[0]-self.patch_sz[0])
            else :
                x = (patch.shape[1]-self.patch_sz[1]) // 2
                y = (patch.shape[0]-self.patch_sz[0]) // 2
            patch = patch[y:y+self.patch_sz[0], x:x+self.patch_sz[1]]

            # Augment train set with rotations and flops
            if self.corrupt:
                patch = np.fliplr(patch) if np.random.randint(2) else patch
                patch = np.rot90(patch, np.random.randint(4))

            images[i] = patch.transpose(2,0,1)
            labels[i] = example.label

            distorsiontime += time()-st

        self.resultqueue.put((images, labels))
        self.distortiontimequeue.put(distorsiontime)
        self.readtimequeue.put(readtime)

    @property
    def shape(self):
        return (1, self.patch_sz[2], self.patch_sz[0], self.patch_sz[1])


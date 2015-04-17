# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:51:42 2015

@author: francis
"""

import random
import numpy            as np
import multiprocessing  as mp
from time               import time
from scipy.misc         import imresize
from liblearn.iterators import iteratorbase
from libutils.function  import addstatic

class randompatch(iteratorbase):

    def __init__(self, examples, patch_sz, reshape_sz, corrupt, load_ratio, dataset_per_epoch=1, nb_workers=3):
        self.examples = examples
        self.patch_sz = patch_sz
        self.reshape_sz = reshape_sz
        self.corrupt = corrupt
        self.load_ratio = load_ratio
        self.dataset_per_epoch = dataset_per_epoch
        iteratorbase.__init__(self, nb_workers)


    @addstatic(readtime=0., distorttime=0.)
    def extractor(self, examples):

        images = np.empty((len(examples),)+self.shape[1:], np.float32)
        labels = np.empty(len(examples), np.int32)

        for i, example in enumerate(examples):

            # read image from disk
            st = time()
            patch = example.get_image()
            self.extractor.statics.readtime += time()-st

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

            self.extractor.statics.distorttime += time()-st

        return (images, labels)

    def maketasks(self):

        # Copy example list
        examples = self.examples[:]

        # Shuffle examples
        random.shuffle(examples)

        # Initialize task queue
        tasks = mp.Queue()

        # Number of examples per task
        task_sz = int(self.load_ratio*len(examples))

        # Number of examples per epoch
        epoch_sz = int(self.dataset_per_epoch*len(examples))

        # Make tasks
        for i in range(0, epoch_sz, task_sz):
            tasks.put(examples[i:i+task_sz])

        return tasks


    @property
    def shape(self):
        return (1, self.patch_sz[2], self.patch_sz[0], self.patch_sz[1])


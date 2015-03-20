# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 13:23:28 2015
@author: francis
"""

if __name__ == '__main__':
    import sys
    sys.path.append("../")

import random
import numpy                   as np
import multiprocessing         as mp
import Queue
from time                      import time
from math                      import ceil
from threading                 import Thread
from scipy.misc                import imresize
from source_                   import source


class features(object):

    def __init__(self, io, load_ratio, patch_sz, reshape_sz):
        assert(isinstance(io, source))
        self.io = io
        self.patch_sz = patch_sz
        self.reshape_sz = reshape_sz
        self.load_ratio = load_ratio


    def train(self):
        split_id = 'train'
        return iterator(self.patch_sz, self.reshape_sz, split_id == 'train', self.io.splits[split_id], self.load_ratio)


    def valid(self):
        split_id = 'valid'
        return iterator(self.patch_sz, self.reshape_sz, split_id == 'train', self.io.splits[split_id], self.load_ratio)

    
    def test(self):
        split_id = 'test'
        return iterator(self.patch_sz, self.reshape_sz, split_id == 'train', self.io.splits[split_id], self.load_ratio)

    
    def shape(self, split_id='train'):
        nb_images = len(self.io.splits[split_id])
        nb_examples = int(ceil(self.load_ratio * nb_images))
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




class iterator(object):
    def __init__(self, patch_sz, reshape_sz, corrupt, examples, load_ratio, nb_workers=6, cashe_sz=2):
        self.tasks = Queue.Queue()
        self.results = Queue.Queue(cashe_sz)
        self.extractor = extractor
        self.load_ratio = load_ratio
        self.examples = list(examples)
        self.nb_workers = nb_workers
        self.patch_sz = patch_sz
        self.reshape_sz = reshape_sz
        self.corrupt = corrupt
        
        # Performance information
        self.__readtime = 0
        self.__distortiontime = 0
        
        # Shuffle examples
        random.shuffle(self.examples)
        
        # Enqueue tasks
        batch_sz = int(self.load_ratio * len(self.examples))
        for batch, i in enumerate(range(0, len(self.examples), batch_sz)):
            self.tasks.put(self.examples[i:i+batch_sz])
        self.__nb_batches = batch+1

        self.__it = 0

        # Start extraction threads
        for i in range(nb_workers):
            worker = Thread(target=self.__worker)
            worker.setDaemon(True)
            worker.start()


    def __iter__(self):
        return self
        
        
    def next(self):
        
        if self.__it < self.__nb_batches:
            self.__it += 1
            return self.results.get()
        raise StopIteration
        

    def __worker(self):
        
        while True:
            
            try:
                examples = self.tasks.get(timeout=0.01)
            except Queue.Empty:
                break

            # Starting extraction process
            process = extractor(examples, self.patch_sz, self.reshape_sz, self.corrupt)
            process.daemon=True
            process.start()
            
            # Extracting result from extractor
            result = process.resultqueue.get()
            
            # Putting result on result queue
            self.results.put(result)
            
            # Wait for process to finish
            process.join()
            
            # Add performance timings
            self.__readtime += process.readtimequeue.get()
            self.__distortiontime += process.distortiontimequeue.get()
            
                

    
    @property
    def readtime(self):
        return self.__readtime

    @property    
    def distortiontime(self):
        return self.__distortiontime




if __name__ == '__main__':
    from os.path import join
    import datasets
    src = source(join(datasets.path(), 'catsanddogs'))

    ft = features(src, 0.1, (100,100, 3), (150,150,3))
   
    st = time()
    it = ft.train()
    for i, example in enumerate(it):
        print 'iteration {}'.format(i)
    
    print 'The train dataset was consumed in {} iterations, taking {} sec'.format(i+1, time()-st)
    print '   Read time : {} sec'.format(it.readtime)
    print '   Distortion time : {} sec'.format(it.distortiontime)
    print 'Done!'
    
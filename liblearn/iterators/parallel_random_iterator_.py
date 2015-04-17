# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 15:52:32 2015

@author: francis
"""

import random
import Queue
from threading                 import Thread


class parallel_random_iterator(object):
    def __init__(self, examples, extractor, extractor_kwargs, load_ratio, dataset_per_epoch, nb_workers=3):
        self.tasks = Queue.Queue()
        self.results = Queue.Queue(2)
        self.load_ratio = load_ratio
        self.dataset_per_epoch = dataset_per_epoch
        self.examples = list(examples)
        self.extractor = extractor
        self.extractor_kwargs = extractor_kwargs

        # Performance information
        self.__readtime = 0
        self.__distortiontime = 0

        # Shuffle examples
        random.shuffle(self.examples)

        # Enqueue tasks
        batch_sz = int(load_ratio * len(self.examples))
        for pos in range(0, int(len(self.examples)*self.dataset_per_epoch), batch_sz):
            self.tasks.put(self.examples[pos:pos+batch_sz])
        self.__nb_batches = self.tasks.qsize()

        self.__it = 0

        # Start extraction threads
        for i in range(nb_workers):
            worker = Thread(target=self.__worker)
            worker.setDaemon(True)
            worker.start()


    def __iter__(self):
        return self


    def __next__(self):

        if self.__it < self.__nb_batches:
            self.__it += 1
            return self.results.get()
        raise StopIteration


    def __worker(self):

        while True:

            try:
                examples = self.tasks.get(timeout=0.01)
            except queue.Empty:
                break

            # Starting extraction process
            process = self.extractor(examples=examples, **self.extractor_kwargs)
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

            del process


    @property
    def readtime(self):
        return self.__readtime

    @property
    def distortiontime(self):
        return self.__distortiontime


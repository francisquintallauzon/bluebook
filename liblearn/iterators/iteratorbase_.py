# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:32:39 2015

@author: francis
"""


import numpy            as np
import multiprocessing  as mp
import Queue            as q
from threading          import Thread
from libutils.dict      import dd


class iteratorbase(object):

    def __init__(self, nb_workers):

        self.__nb_workers = nb_workers
        self.__generators = []
        self.__perftraces = []

    def __iter__(self):

        if not self.__generators:
            self.__generators.append(self.__generate())
        gen = self.__generators.pop(0)
        self.__generators.append(self.__generate())
        return gen

    def __generate(self):

        tasks = self.maketasks()
        results = mp.Queue(1)
        buff = q.Queue(1)
        nb_tasks = tasks.qsize()

        perftrace = dd([(kw, mp.Queue()) for kw in self.extractor.statics])
        self.__perftraces.append(perftrace)

        # Start workers
        processes = []
        for i in range(self.__nb_workers):
            p = mp.Process(target=self.worker, args=(tasks, results), kwargs=perftrace)
            p.daemon = True
            p.start()
            processes.append(p)

        # Start
        t = Thread(target = extractfromqueue, args=(results, buff))
        t.daemon = True
        t.start()

        # Generator function
        def generator(buff, nb_tasks):
            for i in range(nb_tasks):
                yield buff.get()

        # Instanciate generator
        return generator(buff, nb_tasks)



    def worker(self, tasks, results, **perftrace):
        import sys
        while not tasks.empty():
            try:
                results.put(self.extractor(tasks.get(0.1)))
            except q.Empty:
                break
            except :
                print sys.exc_info()
                raise

        # Extracting perftrace information
        for name in perftrace:
            perftrace[name].put(self.extractor.statics[name])


    def extractor(self, task):
        raise NotImplementedError('"extractor" method must be implemented in derived class')


    def maketasks(self):
        raise NotImplementedError('"maketasks" method must be implemented in derived class')


    def getperftrace(self):
        perftrace = self.__perftraces.pop(0)
        for kw in perftrace:
            s = 0.
            while not perftrace[kw].empty():
                s += perftrace[kw].get()
            perftrace[kw] = s
        return perftrace

    def __del__(self):
        print('iteratorbase __del__ called')


def extractfromqueue(src, dst):
    import sys
    while True:
        try:
            dst.put(src.get(0.1))
        except q.Empty:
            pass
        except :
            print sys.exc_info()
            raise



if __name__ == '__main__':
    from time import time, sleep
    from libutils.function import addstatic

    class iterator(iteratorbase):

        @addstatic(elapsed=0.)
        def extractor(self, task):
            st = time()
            sleep
            arr = np.ones((100, 3,100,100))
            self.extractor.statics.elapsed += st-time()
            return arr

        def maketasks(self):
            tasks = mp.Queue()
            for i in range(30):
                tasks.put((i,))
            return tasks

    it = iterator(2)

    for i, data in enumerate(it):
        pass
        #print(i, data.shape)

    print(it.getperftrace())




# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 16:39:38 2013

@author: francis
"""

import os
import pycuda.autoinit
from pycuda.driver          import Device
from os.path                import join, splitext, split, abspath
from subprocess             import Popen as popen
from threading              import Thread
from Queue                  import Queue
from traceback              import print_exc
from time                   import sleep
from datetime               import datetime
from libutils.path    import make_dir

class experiment_scheduler(object):

    def __init__(self, module, experiment_fn, nb_exp = 1, nb_parr=1, device='gpu', device_id = None):
        self.module = module
        self.experiment_fn = experiment_fn
        self.nb_exp = nb_exp
        self.nb_parr = nb_parr
        self.device = device
        self.device_id = device_id
        
    def run(self):

        # Extract experiment name
        experiment_name = splitext(split(self.experiment_fn)[1])[0]

        # Output dir
        self.output_dir = join("../results/", self.module, experiment_name, datetime.now().strftime("%Y%m%d_%Hh%Mm%Ss_%f"))

        # Initialize list of workers (i.e. 1 worker per GPU device)
        self.workers = []

        # Setup a queue of experiments
        self.__tasks = Queue()
        
        # process list
        self.__processes = Queue()

        # Fill experiment queue
        for i in range(self.nb_exp):
            self.__tasks.put((i, join('experiments', self.module, self.experiment_fn)))

        # Start workers
        if self.device_id:
            print 'Running {} experiments on {}{}, (with {} exp / {})'.format(self.nb_exp, self.device, self.device_id, self.nb_parr, self.device)
            print '--------------------------------------------------------------------------\n'
            for i in range(self.nb_parr):
                worker = Thread(target=self.__thread, args=(self.device_id,), name = '{}_{}_thread_{}'.format(self.device, self.device_id, i))
                worker.setDaemon(True)
                worker.start()
                self.workers += [worker]

        else:
            print 'Running {} experiments on {} {}{}, (with {} exp / {})'.format(self.nb_exp, Device.count(), self.device, 's' if self.nb_exp>1 else '', self.nb_parr, self.device)
            print '--------------------------------------------------------------------------\n'
            for device_id in range(Device.count()):
                for i in range(self.nb_parr):
                    worker = Thread(target=self.__thread, args=(device_id,), name = '{}_{}_thread_{}'.format(self.device, self.device_id, i))
                    worker.setDaemon(True)
                    worker.start()
                    self.workers += [worker]
                    sleep(10)

        # Wait until all experiments have finished before returning
        [w.join() for w in self.workers if w.is_alive()]


    def __thread(self, gpu_id):

        while self.__tasks.qsize():

            try:
                exp_id, experiment_fn = self.__tasks.get()
            except Queue.Empty:
                return

            print 'Running experiment {} on gpu{}'.format(exp_id, gpu_id)

            # Make experiment path
            result_path = join(self.output_dir, "{0:05d}".format(exp_id))

            # Make result directory for experiment
            make_dir(result_path)

            try:
                # Make call to experiment file
                call =  ["python", experiment_fn]
                call += ["--path", result_path]

                # Set environment variables
                env = os.environ
                env['THEANO_FLAGS']  = "floatX=float32"
                env['THEANO_FLAGS'] += ",device=gpu{}".format(gpu_id)
                env['THEANO_FLAGS'] += ",nvcc.fastmath=True"
                env['THEANO_FLAGS'] += ",exception_verbosity=high"
                env['THEANO_FLAGS'] += ",optimizer_including=cudnn"
                env['PYTHONPATH']  = abspath("../datasets")
                env['PYTHONPATH'] += ':' + abspath("./")

                # Open process
                p = popen(call, env=env, shell=False)
                self.__processes.put(p)
                
                # Wait for process to finish
                p.wait()
                
            except:
                print_exc()
                with open(join(result_path, 'error.txt'), 'w') as f:
                    print_exc(file=f)


    def kill(self) :
        while self.__processes.qsize():
            p = self.__processes.get()
            print 'Killing process {}'.format(p)
            try :
                p.kill()
            except OSError:
                pass


"""
PRESS F5 TO RUN THIS EXPERIMENT
"""

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Runs experiment scheduler on a python experiment file")
    parser.add_argument('--fn', required=True, type=str, help='Source python file that contains experiment')
    parser.add_argument('--module', required=True, type=str, help='Experiment module (ex. emneuron or bloody)')
    parser.add_argument('--exp', required=False, default=1, type=int, help='Total number of experiments to run')
    parser.add_argument('--parr', required=False, default=1, type=int, help='Number of parallel experiments per device.  1 is recommended for gpu')
    parser.add_argument('--device_type', required=False, default='gpu', type=str, help='Device type to be used, either "cpu" or "gpu"')
    parser.add_argument('--device_id', required=False, default=None, type=int, help='Specific device id gpu to be used.  If none is specified, then all devices are used.  (Does not apply if device==cpu)')

    args = parser.parse_args()
    scheduler = experiment_scheduler(args.module, args.fn, args.exp, args.parr, args.device_type, args.device_id)
    try :
        scheduler.run()
    except  (SystemExit, KeyboardInterrupt) as e:
        print 'Exiting following a SystemExit or KeyboardInterrupt'.format(e.message)
    finally:
        scheduler.kill()
        
    print "Experiment done!"
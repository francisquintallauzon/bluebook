# -*- coding: utf-8 -*-
"""
Implements Metthew Zeiler's ADADELTA adaptive learning rate method
@author: francis
"""


import sys
import warnings

import numpy                as np
import pandas               as pd
import theano               as th
import theano.tensor        as T

from math                   import floor
from math                   import ceil
from time                   import time
from time                   import sleep
from traceback              import print_exception
from traceback              import print_exc
from collections            import OrderedDict
from threading              import Thread as thread
from Queue                  import Queue as queue


class adadelta(object):

    def __init__(self, inp, labels, train_cost, train_error, valid_cost, valid_error, params, maxnorm = 0, debug_calls=None):

        self.debug_call = debug_call

        # Initializations
        self.learning_divergence_counter = 0
        self._epoch_start_time = 0
        self._epoch_valid_time = 0
        self._epoch_updatetrain_time = 0
        self._epoch_learn_time = 0
        self._epoch_loadbatch_time = 0

        # Determine whether learning is supervised or not
        mode = {0:'unsupervised', 1:'supervised'}
        print '    Initializing adadelta instance in {} mode'.format(mode[labels != None])
        self.supervised = labels != None

        # Outputs
        self.cost_train = []
        self.cost_train_smooth = []
        self.cost_valid = []
        self.cost_valid_smooth = []
        self.error_train = []
        self.error_valid = []

        # Validation data
        self.valid_list = []
        self.valid_label_list = []

        # Input parameters
        index = T.lscalar(name = 'lscalar_index')
        minibatch_sz = T.lscalar(name="lscalar_minibatch_sz")
        momentum = T.fscalar(name="fscalar_momentum")
        constant = T.fscalar(name="fscalar_constant")

         # Compute gradient
        self.grads = T.grad(cost=train_cost, wrt=params, disconnected_inputs='ignore', return_disconnected='None')

        # Keep only gradients that are connected to the update tree
        self.grads = OrderedDict([(param, grad) for (param, grad) in zip(params, self.grads) if grad != None])

        # Save a copy of updated parameters
        self.params = self.grads.keys()

        # Get filter maximum norm
        self._learn_maxnorm(maxnorm)

        # Initialize
        self.sq_step_ema = OrderedDict()
        self.sq_grad_ema = OrderedDict()
        for param in self.params:
            self.sq_step_ema[param] = th.shared(np.zeros_like(param.get_value()))
            self.sq_grad_ema[param] = th.shared(np.zeros_like(param.get_value()))


        # Adaptive gradient updates
        updates = OrderedDict()
        for param, grad in self.grads.items():
            sq_grad_ema = momentum * self.sq_grad_ema[param] + (1-momentum) * grad**2
            step_rms = T.sqrt(self.sq_step_ema[param] + constant)
            grad_rms = T.sqrt(sq_grad_ema + constant)
            step = - step_rms / grad_rms * grad
            sq_step_ema = momentum * self.sq_step_ema[param] + (1-momentum) * step**2
            updates[param] = param + step
            updates[self.sq_grad_ema[param]] = sq_grad_ema
            updates[self.sq_step_ema[param]] = sq_step_ema


        # Input shared variable
        self.train = th.shared(np.empty([0]*inp.ndim, inp.dtype), name='training_set')

        # Learning function
        if self.supervised:

            # Output shared variable
            self.train_labels = th.shared(np.empty([0]*labels.ndims, labels.dtype), name='training_set_labels')

            # Theano learning function
            self.learningstep_fn = th.function(inputs = [index, minibatch_sz, momentum, constant],
                                               outputs = [train_cost, train_error],
                                               updates = updates,
                                               givens = {inp: self.train[index*minibatch_sz : (index + 1)*minibatch_sz],
                                                         labels: self.train_labels[index*minibatch_sz : (index + 1)*minibatch_sz]},
                                               allow_input_downcast=True)

            # Clean reconstruction cost (for validation)
            self.valid_cost_fn = th.function(inputs = [inp, labels], outputs = [valid_cost, valid_error])

        else:

            # Theano learning function
            self.learningstep_fn = th.function(inputs = [index, minibatch_sz, momentum, constant],
                                               outputs = train_cost,
                                               updates = updates,
                                               givens = {inp: self.train[index*minibatch_sz : (index + 1)*minibatch_sz]},
                                               allow_input_downcast=True)

            # Clean reconstruction cost (for validation)
            self.valid_cost_fn = th.function(inputs = [inp], outputs = valid_cost)





    def learn(self, lookback, max_epoch, minibatch_sz, train_batch_size_ratio, valid_batch_size_ratio, constant, momentum, fetcher, fetchonce=True, do_validation=True):

        print '    lookback = {}'.format(lookback)
        print '    max_epoch = {}'.format(max_epoch)
        print '    minibatch_sz = {}'.format(minibatch_sz)
        print '    train_batch_size_ratio = {}'.format(train_batch_size_ratio)
        print '    valid_batch_size_ratio = {}'.format(valid_batch_size_ratio)
        print '    constant = {}'.format(constant)
        print '    momentum = {}'.format(momentum)
        print '    fetchonce = {}'.format(fetchonce)
        print '    do_validation = {}'.format(do_validation)

        # Initializations
        self._queue = queue()
        self._done = False
        self._constant = constant
        self._lookback = int(lookback)
        self._minibatch_sz = int(minibatch_sz)
        self._momentum = momentum
        self._fetcher = fetcher
        self._train_batch_size_ratio = train_batch_size_ratio
        self._valid_batch_size_ratio = valid_batch_size_ratio
        self._do_validation = do_validation


        # Fetch initial training and valitation data
        self._load()

        # Update train shared variables
        self._update()

        # Perform a validation step to get the validation loss with random weights (before any training)
        if len(self.cost_train) == 0:
            self._maxnorm()
            self.cost_train = [float('nan')]
            self.error_train = [float('nan')]
            self.cost_train_smooth = [float('nan')]
            if self._do_validation:
                self._valid()
                self._display()

        # Start training
        start = len(self.cost_train)
        end = start + max_epoch
        for i in range(start, end):

            # Set epoch start time
            self._epoch_start_time = time()

            # Update training with data fetch from another memory location (in case the full dataset cannot be stored in memory)
            proc = None
            if fetchonce==False:
                proc = thread(target = self._load, args=(), name = '_load {}'.format(i))
                proc.start()

            # Do one training epoch, concurently with fetching new data
            self._train()

            # Apply maxnorm
            self._maxnorm()

            # Make sure all threads have returned before going to the next iteration
            if proc != None:
                if proc.is_alive():
                    proc.join()

            # Update train set shared variable if needed
            if fetchonce == False:
                self._update()

            # Perform validation step
            if self._do_validation:
                self._valid()

            # Check parameters for divergence
            self._check_params()

            # Call debug function if any
            if self.debug_call != None:
                self.debug_call()

            # Check for errors
            if not self._queue.empty():
                exc_info = self._queue.get()
                print_exception(exc_info[0], exc_info[1], exc_info[2])
                raise exc_info[0], exc_info[1], exc_info[2]

            # Check that training is not yet finished
            if self._done :
                break

            # Print epoch information
            self._display()

        # Reset momentum (usefull if subsequent learning steps are taken )
        #for param, shared in self.last_update.items():
        #    shared.set_value(np.zeros_like(shared.get_value()))

    def _load(self):
        start_time_loadbatch = time()
        try :
            self.train_batch, self.train_batch_labels = self._fetcher('train', self._train_batch_size_ratio)
            self.valid_batch, self.valid_batch_labels = self._fetcher('valid', self._valid_batch_size_ratio)
        except :
            print sys.exc_info()
            self._queue.put(sys.exc_info())
        self._epoch_loadbatch_time = time() - start_time_loadbatch


    def _train(self):
        start_time_learn = time()
        try :
            # Initializations
            cost = 0
            error = 0
            nb_minibatch = int(floor(self.train.get_value(borrow=True).shape[0] / self._minibatch_sz))

            # Backup params
            self.backup_params = {}
            self.backup_params.update({param:param.get_value(borrow=False) for param in self.params})
            self.backup_params.update({param:param.get_value(borrow=False) for param in self.sq_grad_ema.values()})
            self.backup_params.update({param:param.get_value(borrow=False) for param in self.sq_step_ema.values()})

            # Perform minibatch learning
            for i,j in enumerate(np.random.permutation(nb_minibatch)):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if self.supervised:
                        this_cost, this_error = self.learningstep_fn(j, self._minibatch_sz, self._momentum, self._constant)
                        cost += this_cost
                        error += this_error
                    else:
                        this_cost = self.learningstep_fn(j, self._minibatch_sz, self._momentum, self._constant)
                        cost += this_cost

            if self.supervised:
                self.cost_train += [cost / nb_minibatch]
                self.error_train += [error / nb_minibatch]
            else:
                self.cost_train += [cost / nb_minibatch]


            # Smooth cost
            self.cost_train_smooth = pd.stats.moments.ewma(np.asarray(self.cost_train), self._lookback)

        except :
            print_exc()
            self._queue.put(sys.exc_info())
        self._epoch_learn_time = time() - start_time_learn


    def _update(self):
        start_time_updatetrain = time()
        try:
            self.train.set_value(self.train_batch)
            if self.supervised :
                self.train_labels.set_value(self.train_batch_labels)
        except:
            print_exc()
            self._queue.put(sys.exc_info())
        self._epoch_updatetrain_time = time() - start_time_updatetrain


    def _valid(self):
        valid_time_start = time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self.supervised:
                    cost, error = self.valid_cost_fn(self.valid_batch, self.valid_batch_labels)
                    self.cost_valid += [np.float(cost)]
                    self.error_valid += [np.float(error)]
                else:
                    cost = self.valid_cost_fn(self.valid_batch)
                    self.cost_valid += [np.float(cost)]

            # Stop criterion
            self.cost_valid_smooth = pd.stats.moments.ewma(np.asarray(self.cost_valid), self._lookback)

            if len(self.cost_valid) > self._lookback+1:

                # Progress is stopped
                if self.cost_valid_smooth[-self._lookback] < self.cost_valid_smooth[-1] :
                    self._done = True

                # Validation cost > train cost ==> Overfit
                if self.cost_valid_smooth[-1] > self.cost_train_smooth[-1] :
                    self._done = True

        except:
            print_exc()
            self._queue.put(sys.exc_info())
        self._epoch_valid_time = time() - valid_time_start


    def _display(self):
        try :
            total_time = time() - self._epoch_start_time
            ind = len(self.cost_train)-1
            cost_train_str = 'train cost = {:.6f}'.format(self.cost_train[-1])
            cost_train_smooth_str = 'smoothed = {:.6f}'.format(self.cost_train_smooth[-1])
            error_train_str = 'error = {:.6f}'.format(self.error_train[-1]) if self.supervised else ''
            print 'Epoch {}: {}; {}; {}'.format(ind, cost_train_str, cost_train_smooth_str, error_train_str)
            if self._do_validation:
                cost_valid_str = 'valid cost = {:.6f}'.format(self.cost_valid[-1])
                cost_valid_smooth_str = 'smoothed = {:.6f}'.format(self.cost_valid_smooth[-1])
                error_valid_str = 'error = {:.6f}'.format(self.error_valid[-1]) if self.supervised else ''
                print '           {}; {}; {}'.format(cost_valid_str, cost_valid_smooth_str, error_valid_str)
            print '           Time = {:0.2f}s. (Load = {:0.2f}s. Train={:0.2f}s. Valid={:0.2f}s.)'.format(total_time, self._epoch_loadbatch_time, self._epoch_learn_time, self._epoch_valid_time)
            for param in self.params:
                sq_upd = self.sq_step_ema[param].get_value(borrow=True)
                par = param.get_value(borrow=True)
                if sq_upd.ndim == 1:
                    # Bias, not interested
                    continue
                elif sq_upd.ndim == 2:
                    # Regular neural network
                    upd_norm = np.sqrt(sq_upd.sum(0))
                    par_norm = np.sqrt((par**2).sum(0))
                elif sq_upd.ndim == 4 :
                    # Convnet
                    upd_norm = np.sqrt(sq_upd.sum((1,2,3)))
                    par_norm = np.sqrt((par**2).sum((1,2,3)))
                print '           {} : max upd = {}; max mag {}'.format(param, upd_norm.max(), par_norm.max())
        except :
            print_exc()
            print 'Warning : error in printing epoch {} results'.format(ind)
            print 'len(self.cost_valid) = {}'.format(len(self.cost_valid))
            print 'len(self.cost_valid_smooth) = {}'.format(len(self.cost_valid_smooth))
            print 'len(self.cost_train) = {}'.format(len(self.cost_train))
            print 'len(self.cost_train_smooth) = {}'.format(len(self.cost_train_smooth))
            print 'len(self.error_train) = {}'.format(len(self.error_train))
            print 'len(self.error_valid) = {}'.format(len(self.error_valid))


    def _check_params(self):

        # Check for divergence
        revert = False
        #if np.isnan(self.cost_train[-1]):
        #    revert = True
        if len(self.cost_train) > 1:
            if np.isnan(self.cost_train[-1]):
                revert = True
            if self.cost_train[-1] > self.cost_train[-2] * 7.5:
                revert = True

        # Revert parameters if divergence is found
        if revert:

            self.learning_divergence_counter += 1

            if self.learning_divergence_counter >= 25:
                raise ValueError, 'Numerical instability in unsupervised learning at epochs {}'.format(self.learning_divergence_counter)

            if len(self.cost_train) > 1:
                self.cost_train[-1] = self.cost_train[-2]

            if len(self.cost_valid) > 1:
                self.cost_valid[-1] = self.cost_valid[-2]

            # Revert to backup parameter from last epoch
            for param, value in self.backup_params.items():
                param.set_value(value, borrow=False)

            # Reset momentum
            #for param, shared in self.last_update.items():
            #    shared.set_value(np.zeros_like(shared.get_value()))

            # Reduce learning rate
            self._constant /= 10.
            print "Numerical instability at epoch {}.  Set intialization constant to {:.10f} (from {:.10f}) ".format(len(self.cost_train)-1, self._constant, self._constant*10.)

            return

        # No divergence, then, reset the learning divergences counter
        self.learning_divergence_counter = 0


    def _learn_maxnorm(self, maxnorm):
        if maxnorm <=0 :
            self._maxnormd=None
            return

        self._maxnormd = OrderedDict()
        for p in self.params:
            if "Logistic" in p.name or "logistic" in p.name:
                continue
            q = p.get_value(borrow=True)
            if q.ndim == 2:
                # Regular neural network
                norm = np.sqrt((q**2).sum(0))
                self._maxnormd[p] = norm.max() * maxnorm
            elif q.ndim == 4 :
                # Convnet
                norm = np.sqrt((q**2).sum((1,2,3)))
                self._maxnormd[p] = norm.max() * maxnorm

        for key, val in self._maxnormd.items():
            print "    maxnorm for {} = {}".format(key, val)


    def _maxnorm(self):
        if self._maxnormd == None:
            return

        for p in self._maxnormd.keys():
            q = p.get_value(borrow=True)
            maxnorm = self._maxnormd[p]
            if q.ndim == 2:
                norm = np.sqrt((q**2).sum(0))           # Regular neural network
            elif q.ndim == 4 :
                norm = np.sqrt((q**2).sum((1,2,3)))     # Convnet
            if norm.max() > maxnorm:
                q = q * maxnorm / norm.max()
            p.set_value(q)


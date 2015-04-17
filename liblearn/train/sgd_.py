# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 21:53:09 2013
@author: francis
"""

if __name__ == '__main__':
    import sys
    sys.path += ['../../']

import gc
import sys
import warnings

import numpy                    as np
import pandas                   as pd
import theano                   as th
import theano.tensor            as T

from os.path                    import join
from math                       import exp, log, sqrt
from time                       import time

from traceback                  import print_exc
from libutils.dict              import dd
from libutils.matplotlib        import subplots
from libutils.function          import timing, propertydescriptor
from liblearn.utils             import cast_x, shared_x, shared_copy, shared_zeros_like

class sgd(object):

    train_data = propertydescriptor()
    valid_data = propertydescriptor()

    def __init__(self, inp, train_data, valid_data=None,
                       max_epoch=10000, lookback=1000, minibatch_sz=100,
                       init_lr = None, incr_lr = None, lr=0.01, decay_rate=0,
                       momentum=0.9, momentum_reset_prob = 0,
                       loops_per_epoch = 1, output_path=None, log=dd()):

        """
        Implement stochastic gradient descent

        Parameters
        ----------
        inp:            theano tensor
                        model input

        train_data / valid_data:     tuple or function object
                        training/validation data with (examples, labels)
                        examples should be a ndarray with shape [nb_examples, dim1, dim2, ...]
                        labels should be None for unsupervised learning or a ndarray with shape [nb_examples]
                        If train_data is a function object, a call without argument should return a tuple.

        max_epoch:      integer type
                        Maximum number epochs

        lookback:       integer type
                        EMA smoothing period on validation cost, used for
                        early-stop, to prevent overfit

        minibatch_sz:   integer type
                        number of examples per parameter update

        init_lr:        float type
                        Initial learning rate

        incr_lr:        float type
                        Learning rate increment applied after each epoch until
                        init_lr + lr_increment >= lr.  Then the learning rate
                        lr is decayed.

        lr:             float type
                        Gradient descent's learning rate

        decay_rate:     float_type
                        Decay rate
                        learning_rate (t+1) = learning_rate(t) * decay_rate

        momentum:       float type
                        last_update*momentum + learning_rate*grad

        momentum_reset_prob: float type
                        probability of reseting momentum after on on an epoch

        log :           dict like object
                        Dictionary in which the trainer will log learning's
                        current status.

        output_path :   str
                        output path to which the trainer will dump the log dict
                        to text file and ouput a graph of the learning
                        progression


        """

        # Model input
        self.inp = inp
        self.max_epoch = max_epoch
        self.lookback = lookback
        self.minibatch_sz = int(minibatch_sz)
        self.init_lr = init_lr
        self.incr_lr = incr_lr
        self.high_lr = lr
        self.decay_rate = decay_rate
        self.momentum = momentum
        self.train_data = train_data
        self.valid_data = valid_data
        self.log = log
        self.output_path = output_path
        self.subplots = dd()
        self.do_validation = valid_data is not None
        self.momentum_reset_prob = momentum_reset_prob
        self.loops_per_epoch = loops_per_epoch

        # Learning data shared variables
        self.shr_train_data = shared_x(np.empty([0]*self.inp.ndim), name='training_set_data')


    def __call__(self, params, train_cost, valid_cost=None, labels=None, train_error=None, valid_error=None,  model_id='',
                 additionnal_updates=[], debug_calls=[], debug_nodes={}):

        # Determine whether learning is supervised or not
        self.supervised = labels != None
        self.do_validation = self.do_validation and bool(valid_cost)
        print('\nInitializing sgd for {} learning of {} {} validation.'.format({0:'unsupervised', 1:'supervised'}[self.supervised], model_id, {0:'without', 1:'with'}[self.do_validation]))

        # Initializations
        self.additionnal_updates= additionnal_updates if isinstance(additionnal_updates, list) else [additionnal_updates]
        self.learning_divergence_counter = 0
        self.rampup = bool(self.init_lr)
        self.lr = self.init_lr if self.rampup else self.high_lr

        # Outputs
        self.cost = dd()
        self.cost.epoch = []
        self.cost.train = []
        self.cost.train_ema = []
        self.cost.valid = []
        self.cost.valid_ema = []

        self.error = dd()
        self.error.epoch = []
        self.error.train = []
        self.error.train_ema = []
        self.error.valid = []
        self.error.valid_ema = []

        # Validation data
        self.valid_list = []
        self.valid_label_list = []

        # Input parameters
        index = T.lscalar(name = 'lscalar_index')
        minibatch_sz = T.lscalar(name="lscalar_minibatch_sz")
        momentum = T.fscalar(name="fscalar_momentum")
        learning_rate = T.fscalar(name="fscalar_learning_rate")

        # Save reference to parameters
        self.params = params

         # Compute gradient
        self.grads = T.grad(cost=train_cost, wrt=params, disconnected_inputs='ignore', return_disconnected='None')

        # Keep only gradients that are connected to the update tree
        self.grads = dd([(param, grad) for (param, grad) in zip(params, self.grads) if grad != None])

        # Save initial value of parameters
        self.init = dd([(param, shared_x(param.get_value())) for param in self.params])

        if self.do_validation:
            self.best_performance = sys.float_info.max
            self.best_params = dd([(param, param.get_value()) for param in self.params])

        # Initialize lastupdates
        self.last_update = dd([(p, shared_x(np.zeros_like(p.get_value()))) for p in self.params])

        # Learning updates
        updates = []
        for param, grad in list(self.grads.items()):
            last = self.last_update[param]
            gradient = last*momentum + learning_rate*grad
            updates.append((param, param-gradient))
            updates.append((last, gradient))

        # Learning function
        if self.supervised:

            # Learning labels shared variables
            self.shr_train_labels = th.shared(np.empty([0]*labels.ndim, dtype=labels.dtype), name='training_set_labels')

            # Learning function
            self.learningstep_fn = th.function(inputs = [index, minibatch_sz, momentum, learning_rate],
                                               outputs = [train_cost, train_error],
                                               updates = updates,
                                               givens = {self.inp: self.shr_train_data[index*minibatch_sz : (index + 1)*minibatch_sz],
                                                           labels: self.shr_train_labels[index*minibatch_sz : (index + 1)*minibatch_sz]},
                                               allow_input_downcast=True)

            # Noisy cost on train shared variables
            self.train_cost_fn = th.function(inputs = [], outputs = [valid_cost, valid_error], givens = {self.inp: self.shr_train_data, labels: self.shr_train_labels}, allow_input_downcast=True)

            # Clean reconstruction cost (for validation)
            self.valid_cost_fn = th.function(inputs = [self.inp, labels], outputs = [valid_cost, valid_error], allow_input_downcast=True)

            if debug_nodes:
                self.debug_fn = th.function(inputs = [], outputs = [output for output in list(debug_nodes.values())],
                                            givens = {self.inp: self.shr_train_data, labels:self.shr_train_labels},
                                            allow_input_downcast=True, on_unused_input='ignore')

        else:
            # Theano learning function
            self.learningstep_fn = th.function(inputs = [index, minibatch_sz, momentum, learning_rate],
                                               outputs = train_cost,
                                               updates = updates,
                                               givens = {self.inp: self.shr_train_data[index*minibatch_sz : (index + 1)*minibatch_sz]},
                                               allow_input_downcast=True, on_unused_input='ignore')


            # Noisy cost on train shared variables
            self.train_cost_fn = th.function(inputs = [], outputs = valid_cost, givens = {self.inp: self.shr_train_data}, allow_input_downcast=True)

            # Clean reconstruction cost (for validation)
            self.valid_cost_fn = th.function(inputs = [self.inp], outputs = valid_cost, allow_input_downcast=True)

            if debug_nodes:
                self.debug_fn = th.function(inputs = [], outputs = [output for output in list(debug_nodes.values())],
                                            givens = {self.inp: self.shr_train_data},
                                            allow_input_downcast=True, on_unused_input='ignore')

        # Debug
        self.debug_calls = debug_calls
        self.debug_nodes = dd(debug_nodes)
        self.model_id = model_id

        # For learning stats graph outputs
        self.last_batch_param = dd([(param, shared_copy(param)) for param in self.params])
        self.last_batch_update = dd([(param, shared_zeros_like(param)) for param in self.params])
        self.this_batch_update = dd([(param, shared_zeros_like(param)) for param in self.params])

        updates = []
        for param in self.params:
            updates += [(self.last_batch_update[param], self.this_batch_update[param])]
            updates += [(self.this_batch_update[param], param - self.last_batch_param[param])]
            updates += [(self.last_batch_param[param], param)]
        self.update_learning_stats_fn = th.function(inputs=[], outputs=[], updates=updates)

        # Initialize learning stats plots
        if self.subplots:
            for sp in list(self.subplots.values()):
                sp.clf()
            self.subplots = dd()

        self.subplots.graphs = subplots(1, 1+self.supervised, 3, 3+3*self.supervised, projection='recurrent')

        line_names = ['p005', 'median', 'p995', 'std']
        line_labels = ['0.5%', 'median', '99.5%', 'std']
        for param in self.params:
            nb_plots = 3 if param.ndim == 1 else 6
            sp = subplots(2, nb_plots, 6, nb_plots*3, projection='recurrent')
            for i in range(nb_plots):
                for name, label in zip(line_names, line_labels):
                    sp[1,i].add_line(name=name, label=label)
                sp[1,i].set(xlabel='epoch', xscale='log', xtick_fontsize=6, ytick_fontsize=6)
                sp[1,i].legend(loc='upper center', fontsize=6)
                sp[0,i].set(xtick_fontsize=6, ytick_fontsize=6)
            if nb_plots == 6:
                sp[1,3].set(yscale='log')
            self.subplots[param] = sp

        for node in list(self.debug_nodes.values()):
            sp = subplots(2, 1, 6, 3, projection='recurrent')
            for name, label in zip(line_names, line_labels):
                sp[1,0].add_line(name=name, label=label)
            sp[1,0].add_line(name='nonzero', label='non-zeros')
            sp[1,0].set(xlabel='epoch', xscale='log', xtick_fontsize=6, ytick_fontsize=6)
            sp[1,0].legend(loc='upper center', fontsize=6)
            sp[0,0].set(xtick_fontsize=6, ytick_fontsize=6)
            self.subplots[node] = sp

        self.subplots.graphs[0,0].set(xlabel='epoch', yscale='log', xtick_fontsize=8, ytick_fontsize=8)
        self.subplots.graphs[0,0].set_title('cost' ' - {}'.format(model_id) if model_id else None, fontsize=10)
        self.subplots.graphs[0,0].add_line(name='train', label = 'train')
        self.subplots.graphs[0,0].add_line(name='train_ema', label = 'train (EMA)')
        if self.do_validation:
            self.subplots.graphs[0,0].add_line(name='valid', label = 'valid')
            self.subplots.graphs[0,0].add_line(name='valid_ema', label = 'valid (EMA)')
        self.subplots.graphs[0,0].legend(loc='best', fontsize=6)

        if self.supervised:
            self.subplots.graphs[0,1].set(xlabel='epoch', yscale='log', xtick_fontsize=8, ytick_fontsize=8)
            self.subplots.graphs[0,1].set_title('error', fontsize=10)
            self.subplots.graphs[0,1].add_line(name='train', label = 'train')
            self.subplots.graphs[0,1].add_line(name='train_ema', label = 'train (EMA)')
            if self.do_validation:
                self.subplots.graphs[0,1].add_line(name='valid', label = 'valid')
                self.subplots.graphs[0,1].add_line(name='valid_ema', label = 'valid (EMA)')
            self.subplots.graphs[0,1].legend(loc='best', fontsize=6)

        return self


    @timing
    def learn(self):

        # Start training
        start_epoch = self.cost.epoch[-1] if self.cost.epoch else 0
        end_epoch = start_epoch + self.max_epoch
        for epoch in np.arange(start_epoch, end_epoch, self.loops_per_epoch):

            # Set epoch start time
            start_tm = time()

            # Do one training epoch, concurently with fetching new data
            self.__train()

            # Perform validation step
            if self.do_validation:
                self.__valid(epoch)

            # Check for gradient explosion and revert parameters if necessary
            self.__revert_exploding_gradient()

            # Compute smoothed time series
            self.__smooth()

            # Apply additionnal updates
            for upd in self.additionnal_updates:
                upd()

            # File outputs
            self.__log_output(epoch)
            #self.__graph_output(epoch)
            self.__debug()

            # Output basic epoch info to console
            self.__console_output(epoch, time()-start_tm)

            # Change learning rate for next step
            if self.rampup: # Linearly increment learning rate
                self.lr += self.incr_lr
                if self.lr > self.high_lr:
                    self.lr = self.high_lr
                    self.rampup = False
            elif self.decay_rate: # Decay learning rate
                self.lr = exp(log(self.lr) + log(self.decay_rate) * self.loops_per_epoch)

            # Reset momentum
            if np.random.binomial(1, self.momentum_reset_prob):
                self.__reset_momentum()

            # Overfit based stopping criterion
            if self.do_validation:
                if epoch >= self.lookback:
                    try :
                        if self.supervised:
                            if self.error.valid_ema[-1] > self.error.valid_ema[-self.lookback//self.loops_per_epoch-1]:
                                break
                        else :
                            if self.cost.valid_ema[-1] > self.cost.valid_ema[-self.lookback//self.loops_per_epoch-1]:
                                break
                    except :
                        print(self.cost.epoch,  self.lookback+1)
                        print(-self.lookback//self.loops_per_epoch-1)
                        print(len(self.cost.valid_ema))
                        print(len(self.error.valid_ema))
                        raise


            # Stop learning if learning rate becomes too small
            if self.lr / self.high_lr < 1e-10:
                break

        if self.do_validation:
            for param in self.best_params:
                param.set_value(self.best_params[param], borrow=False)



    @timing
    def __train(self):


        # Initializations
        cost = 0
        error = 0
        nb_minibatches = 0
        nb_tasks = 0

        # Backup params
        self.backup_params = {param:param.get_value(borrow=False) for param in self.params}

        setvaltime = 0
        processtime = 0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Extract iterator if available (this is why self.train_data is a property descriptor)
            for i, (data, labels) in enumerate(self.train_data):

                # Update train set shared variable if needed
                st = time()
                self.shr_train_data.set_value(data, borrow=True)
                if self.supervised :
                    self.shr_train_labels.set_value(labels, borrow=True)
                setvaltime += time()-st

                # Perform minibatch learning
                st=time()
                for j in range(data.shape[0] // self.minibatch_sz):
                        output = self.learningstep_fn(j, self.minibatch_sz, self.momentum, self.lr)
                        cost += output[0]
                        if self.supervised :
                            error += output[1]
                        nb_minibatches += 1
                processtime += time()-st

                nb_tasks += 1

            if hasattr(self.train_data, 'getperftrace'):
                print('    Performance on approx {} examples, divided in {} tasks: '.format(nb_minibatches*self.minibatch_sz, nb_tasks))
                print('        Set value time       = {:3.2f}'.format(setvaltime))
                print('        GPU processing time  = {:3.2f}'.format(processtime))
                for name, val in self.train_data.getperftrace().items():
                    print('        Data {:15s} = {:3.2f}'.format(name, val))
                gc.collect()

        if nb_minibatches :
            self.cost.train += [cost/nb_minibatches]
            if self.supervised:
                self.error.train += [error/nb_minibatches]

    @timing
    def __valid(self, epoch):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cost = 0.
            error = 0.
            nb_examples = 0
            for i, (data, labels) in enumerate(self.valid_data):
                if self.supervised:
                    this_cost, this_error = self.valid_cost_fn(data, labels)
                    cost += this_cost * data.shape[0]
                    error += this_error * data.shape[0]
                else:
                    cost += self.valid_cost_fn(data) * data.shape[0]
                nb_examples += data.shape[0]

        self.cost.epoch += [epoch]
        self.cost.valid += [cost / nb_examples]
        if self.supervised:
            self.error.valid += [error / nb_examples]
            self.error.epoch += [epoch]


        performance = self.error.valid[-1] if self.supervised else self.cost.valid[-1]

        if performance < self.best_performance:
            if self.best_performance != sys.float_info.max:
                print('    New best performance achieved at {:0.4f} from {:0.4f}'.format(performance, self.best_performance))
            self.best_performance = performance
            for param in self.best_params:
                self.best_params[param] = param.get_value(borrow=False)


    @timing
    def __console_output(self, epoch, total_time = 0):
        epoch_str = '    Epoch {:10.4f}: '.format(epoch)
        cost_train_str = 'train cost = {:.4f}'.format(self.cost.train[-1])
        cost_train_smooth_str = 'smoothed = {:.4f}'.format(self.cost.train_ema[-1])
        error_train_str = 'error = {:.4f}'.format(self.error.train[-1]) if self.supervised else ''
        print('{}{}; {}; {}'.format(epoch_str, cost_train_str, cost_train_smooth_str, error_train_str))
        offset =' '*len(epoch_str)
        if self.do_validation:
            cost_valid_str = 'valid cost = {:.4f}'.format(self.cost.valid[-1])
            cost_valid_smooth_str = 'smoothed = {:.4f}'.format(self.cost.valid_ema[-1])
            error_valid_str = '\033[1merror = {:.4f}\033[0m'.format(self.error.valid[-1]) if self.supervised else ''
            print('{}{}; {}; {}'.format(offset, cost_valid_str, cost_valid_smooth_str, error_valid_str))
        print('{}Time = {:0.2f}s. (Train={:0.2f}s. Valid={:0.2f}s. Log={:0.2f}s. Graphs={:0.2f}s. Debug={:0.2f}s.)'.format(offset, total_time, self.__train.last, self.__valid.last, self.__log_output.last, self.__graph_output.last, self.__debug.last).rjust(len('Epoch')+10))
        print('{}Learning rate = {}'.format(offset, self.lr))

    @timing
    def __graph_output(self, epoch):

        self.update_learning_stats_fn()

        ########  Learning statistic associated with optimized parameters
        for param in self.params:

            last_update = self.last_batch_update[param]
            this_update = self.this_batch_update[param]
            init = self.init[param]
            sp = self.subplots[param]
            name = str(param)

            if param.ndim == 1:
                data = param.get_value()
                p005, median, p995 = np.percentile(data, [0.5, 50, 99.5])
                sp[0,0].hist(remove=True, x=data, bins=35)
                sp[0,0].set_title('{} at epoch {}'.format(param, epoch), fontsize=10)
                sp[1,0].add_point(p005=(epoch,p005), median=(epoch,median), p995=(epoch,p995), std=(epoch,data.std()))

                data = (param - init).eval()
                p005, median, p995 = np.percentile(data, [0.5, 50, 99.5])
                sp[0,1].hist(remove=True, x=data, bins=35)
                sp[0,1].set_title('{}-initial'.format(param), fontsize=10)
                sp[1,1].add_point(p005=(epoch,p005), median=(epoch,median), p995=(epoch,p995), std=(epoch,data.std()))

                data = this_update.get_value()
                p005, median, p995 = np.percentile(data, [0.5, 50, 99.5])
                sp[0,2].hist(remove=True, x=data, bins=35)
                sp[0,2].set_title('{} gradient update'.format(param), fontsize=10)
                sp[1,2].add_point(p005=(epoch,p005), median=(epoch,median), p995=(epoch,p995), std=(epoch,data.std()))

            elif param.ndim > 1:

                if param.ndim == 2:
                    param = param.T
                    last_update = last_update.T
                    this_update = this_update.T
                    init = init.T

                param = param.flatten(2)
                last_update = last_update.flatten(2)
                this_update = this_update.flatten(2)
                init = init.flatten(2)

                # Norms
                nrm = T.sqrt((param**2).sum(1))
                data = nrm.eval()
                p005, median, p995 = np.percentile(data, [0.5, 50, 99.5])
                sp[0,0].hist(remove=True, x=data, bins=35)
                sp[0,0].set_title(r'$\Vert w_i \Vert \/ i \in [1,{}]$ at epoch {}'.format(len(data), epoch), fontsize=10)
                sp[1,0].add_point(p005=(epoch,p005), median=(epoch,median), p995=(epoch,p995), std=(epoch,data.std()))

                # Orthonormality
                param_nrm = param / nrm[:,None]
                data = T.dot(param_nrm, param_nrm.T).flatten().eval()
                p005, median, p995 = np.percentile(data, [0.5, 50, 99.5])
                sp[0,1].hist(remove=True, x=data, bins=60)
                sp[0,1].set_yscale('log', nonposy='clip')
                sp[0,1].set_title(r'$ {{ \frac{{ {{w_i}}^\intercal w_j }}{{ \Vert w_i \Vert \Vert w_j \Vert }} }} {{\vert}}_{{(t={})}}  \/ i,j \in [1,{}]  $'.format(epoch, int(sqrt(len(data)))), fontsize=10)
                sp[1,1].add_point(p005=(epoch,p005), median=(epoch,median), p995=(epoch,p995), std=(epoch,data.std()))

                # Rotations with respect to initial state
                cos = (param*init).sum(1)
                nrm = T.sqrt((param**2).sum(1))
                nrm_init = T.sqrt((init**2).sum(1))
                fac = cast_x(180. / np.pi)
                data = (T.arccos(cos/(nrm*nrm_init)) * fac).flatten().eval()
                p005, median, p995 = np.percentile(data, [0.5, 50, 99.5])
                sp[0,2].hist(remove=True, x=data, bins=35)
                sp[0,2].set_title(r'$ \measuredangle ( w^{{(t={})}}_i, w^{{(t=0)}}_i ) \/ i \in [1,{}] $'.format(epoch, len(data)), fontsize=10)
                sp[1,2].add_point(p005=(epoch,p005), median=(epoch,median), p995=(epoch,p995), std=(epoch,data.std()))

                # Update norm
                data = T.sqrt((this_update**2).sum(1)).eval()
                p005, median, p995 = np.percentile(data, [0.5, 50, 99.5])
                sp[0,3].hist(remove=True, x=data, bins=35)
                sp[0,3].set_title(r'$\Vert u_i \Vert \/ i \in [1,{}]$ at epoch {}'.format(len(data), epoch), fontsize=10)
                sp[1,3].add_point(p005=(epoch,p005), median=(epoch,median), p995=(epoch,p995), std=(epoch,data.std()))

                # Update rotation with respect to weight vectors
                cos = (param*this_update).sum(1)
                nrm = T.sqrt((param**2).sum(1))
                nrm_init = T.sqrt((this_update**2).sum(1))
                fac = cast_x(180. / np.pi)
                data = (T.arccos(cos/(nrm*nrm_init)) * fac).flatten().eval()
                p005, median, p995 = np.percentile(data, [0.5, 50, 99.5])
                try:
                    sp[0,4].hist(remove=True, x=data, bins=35)
                except:
                    print(param)
                    print(data.shape)
                    raise
                sp[0,4].set_title(r'$ \measuredangle ( w^{{(t={})}}_i, u^{{(t={})}}_i ) \/ i \in [1,{}] $'.format(epoch, epoch, len(data)), fontsize=10)
                sp[1,4].add_point(p005=(epoch,p005), median=(epoch,median), p995=(epoch,p995), std=(epoch,data.std()))

                # Update rotation if this update with respect to the last
                cos = (this_update*last_update).sum(1)
                nrm_this = T.sqrt((last_update**2).sum(1))
                nrm_last = T.sqrt((this_update**2).sum(1))
                fac = cast_x(180. / np.pi)
                data = (T.arccos(cos/(nrm_this*nrm_last)) * fac).flatten().eval()
                p005, median, p995 = np.percentile(data, [0.5, 50, 99.5])
                sp[0,5].hist(remove=True, x=data, bins=35)
                sp[0,5].set_title(r'$ \measuredangle ( u^{{(t={})}}_i, u^{{(t={})}}_i ) \/ i \in [1,{}] $'.format(epoch, epoch-1, len(data)), fontsize=10)
                sp[1,5].add_point(p005=(epoch,p005), median=(epoch,median), p995=(epoch,p995), std=(epoch,data.std()))

            else :
                continue

            sp.savefig(join(self.output_path, '{}_{}_learning_stats_{}.png'.format({0:'unsupervised', 1:'supervised'}[self.supervised], self.model_id, name)), dpi = 100)

        ########  Learning statistic associated with optimized parameters
        if self.debug_nodes:
            outputs = self.debug_fn()
            for (name, node), data in zip(list(self.debug_nodes.items()), outputs):
                sp = self.subplots[node]
                data = data.flatten()
                nonzeros = float((data!=0).mean())
                p005, median, p995 = np.percentile(data, [0.5, 50, 99.5])
                sp[0,0].hist(remove=True, x=data, bins=60)
                sp[0,0].set_yscale('log', nonposy='clip')
                sp[0,0].set_title('{} at t={}'.format(name, epoch), fontsize=6)
                sp[1,0].add_point(p005=(epoch,p005), median=(epoch,median), p995=(epoch,p995), std=(epoch,data.std()),nonzero=(epoch,nonzeros))
                sp[1,0].set_title('Non-zero = {:0.4f}%'.format(nonzeros*100), fontsize=8)
                sp.savefig(join(self.output_path, name+'.png'), dpi = 100)


    def __revert_exploding_gradient(self):

        # Check for divergence
        revert = False
        if len(self.cost.train) > 1:
            if np.isnan(self.cost.train[-1]):
                revert = True
            if self.cost.train[-1] > self.cost.train[-2] * 7.5:
                revert = True

        # Revert parameters if divergence is found
        if revert:

            # Stop rampup
            # self.rampup = False

            self.learning_divergence_counter += 1

            if self.learning_divergence_counter >= 100:
                raise ValueError('    --  Numerical instability in unsupervised learning at epochs {}'.format(self.learning_divergence_counter))

            if len(self.cost.train) > 1:
                self.cost.train[-1] = self.cost.train[-2]

                if self.do_validation:
                    self.cost.valid[-1] = self.cost.valid[-2]

            # Revert to backup parameter from last epoch
            for param, value in list(self.backup_params.items()):
                param.set_value(value, borrow=False)

            # Reset momentum
            self.__reset_momentum()

            if self.learning_divergence_counter > 10:
                self.lr /= 1.1
                print("    --  Numerical instability at epoch {}.  Reseting momentum.  Set learning rate to {}".format(len(self.cost.train)-1, self.lr))
            else:
                print("    --  Numerical instability at epoch {}.  Reseting momentum.".format(len(self.cost.train)-1))

            return True

        # No divergence, then, reset the learning divergences counter
        self.learning_divergence_counter = 0

        return False

    @timing
    def __log_output(self, epoch):

        # UNSUPERVISED ---------------------------------------------------------------------------

        # Set current model log dictionnary
        model_log = dd()

        model_log.nb_epoch = epoch

        if self.cost.valid:
            if "cost" not in model_log:
                model_log.cost = dd()
            model_log.cost.valid = self.cost.valid[-1]
            model_log.cost.valid_smooth = self.cost.valid_ema[-1]

        if self.error.valid:
            if "error" not in model_log:
                model_log.error = dd()
            model_log.error.valid = self.error.valid[-1]
            model_log.error.valid_smooth = self.error.valid_ema[-1]

        if self.cost.train:
            if "cost" not in model_log:
                model_log.cost = dd()
            model_log.cost.train = self.cost.train[-1]
            model_log.cost.train_smooth = self.cost.train_ema[-1]

        if self.error.train:
            if "error" not in model_log:
                model_log.error = dd()
            model_log.error.train = self.error.train[-1]
            model_log.error.train_smooth = self.error.train_ema[-1]

        self.log[self.model_id] = model_log
        self.log.dump(join(self.output_path, "out.pkl"), True)

        if self.cost.train:
            self.subplots.graphs[0,0].add_point(train=(epoch, self.cost.train[-1]), train_ema=(epoch, self.cost.train_ema[-1]))

        if self.cost.valid:
            self.subplots.graphs[0,0].add_point(valid=(epoch, self.cost.valid[-1]), valid_ema=(epoch, self.cost.valid_ema[-1]))

        if self.error.train:
            self.subplots.graphs[0,1].add_point(train=(epoch, self.error.train[-1]), train_ema=(epoch, self.error.train_ema[-1]))

        if self.error.valid:
            self.subplots.graphs[0,1].add_point(valid=(epoch, self.error.valid[-1]), valid_ema=(epoch, self.error.valid_ema[-1]))

        self.subplots.graphs.save(join(self.output_path, self.model_id + ".png"), dpi=100)


    @timing
    def __debug(self):
        # Call debug function if any
        try:
            if self.debug_calls:
                if not isinstance(self.debug_calls, list):
                    self.debug_calls = [self.debug_calls]
                for debug_call in self.debug_calls:
                    debug_call()
        except:
            warnings.warn('Call to debug_call function has thrown the following exception.')
            print_exc()


    def __smooth(self):
        self.cost.train_ema = pd.stats.moments.ewma(np.asarray(self.cost.train), self.lookback / self.loops_per_epoch)
        self.cost.valid_ema = pd.stats.moments.ewma(np.asarray(self.cost.valid), self.lookback / self.loops_per_epoch)
        self.error.train_ema = pd.stats.moments.ewma(np.asarray(self.error.train), self.lookback / self.loops_per_epoch)
        self.error.valid_ema = pd.stats.moments.ewma(np.asarray(self.error.valid), self.lookback / self.loops_per_epoch)


    def __reset_momentum(self):
        # Reset momentum
        for param, shared in list(self.last_update.items()):
            shared.set_value(np.zeros_like(shared.get_value()))


if __name__ == '__main__':

    inp = T.matrix('inp_fmatrix')
    lab = T.ivector('lab_ivector')

    nb_examples = 10000
    train_data = np.random.uniform(-1,1, (nb_examples, 100))
    train_labels = np.random.randint(0, 2, nb_examples).astype(np.int32)
    valid_data = np.random.uniform(-1,1, (1000, 100))
    valid_labels = np.random.randint(0, 2, 1000).astype(np.int32)


    from libutils.path import make_dir
    make_dir('./sgd_test')

    from liblearn.layer import logistic
    from liblearn.loss  import cross_entropy, error

    model = logistic((10000, 100), 2)
    prob = model(inp)
    cost = cross_entropy(prob, lab)
    error = error(prob, lab)


    test = sgd(inp=inp, train_data=[(train_data, train_labels)], valid_data=[(valid_data, valid_labels)],
                      max_epoch=10, lookback=10, minibatch_sz=10, lr=0.01, output_path = './sgd_test')

    test = test(list(model.params.values()), cost, cost, lab, error, error,  model_id='test')

    test.learn()

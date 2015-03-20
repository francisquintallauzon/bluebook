# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 10:25:17 2014

@author: francis
"""

import numpy as np
import theano as th
import theano.tensor as T
from utils.dict                 import dd
from math                       import atan2

class _square_fit_from_corners(object):
    
    def __init__(self):
        
        self.learning_rate_t_fac = 1/10000000.
        
        # Convert corner data into shared variables   
        self.ur_data = th.shared(np.empty((0, 2), th.config.floatX), name = "ur_data", allow_downcast=True)
        self.ul_data = th.shared(np.empty((0, 2), th.config.floatX), name = "ul_data", allow_downcast=True)
        self.ll_data = th.shared(np.empty((0, 2), th.config.floatX), name = "ll_data", allow_downcast=True)
        self.lr_data = th.shared(np.empty((0, 2), th.config.floatX), name = "lr_data", allow_downcast=True)

        # Parameter of square to be fitted to the data
        self.p = th.shared(np.asarray([0,0], th.config.floatX)[None, :], 'center', allow_downcast=True)
        self.t = th.shared(np.asarray(0, th.config.floatX), 'theta', allow_downcast=True)
        self.w = th.shared(np.asarray(0, th.config.floatX), 'width', allow_downcast=True)
        params = [self.p, self.t, self.w]

        # Model corner data 
        self.ur_model = T.stacklists([ self.w/2,  self.w/2])
        self.ul_model = T.stacklists([-self.w/2,  self.w/2])
        self.ll_model = T.stacklists([-self.w/2, -self.w/2])
        self.lr_model = T.stacklists([ self.w/2, -self.w/2])
        
        # Rotation matrix associated to angle
        R = T.stacklists([[T.cos(self.t), -T.sin(self.t)],[T.sin(self.t), T.cos(self.t)]])
            
        # Symbolic cost function
        cost   = ((self.p + T.dot(self.ur_model, R.T) - self.ur_data)**2).sum()
        cost  += ((self.p + T.dot(self.ul_model, R.T) - self.ul_data)**2).sum()
        cost  += ((self.p + T.dot(self.ll_model, R.T) - self.ll_data)**2).sum()
        cost  += ((self.p + T.dot(self.lr_model, R.T) - self.lr_data)**2).sum()
        
        # Hyperparameters
        learning_rate = T.scalar('learning_rate', th.config.floatX)
        
        # Symbolic gradient
        grads = dd([(param, grad) for param, grad in zip(params, T.grad(cost=cost, wrt=params))])
        
        # Symbolic update
        updates = dd([(param, param - (learning_rate*self.learning_rate_t_fac if param==self.t else learning_rate)*grad) for param, grad in grads.items()])
        
        # Theano update function
        self.update_fn = th.function(inputs = [learning_rate], 
                                     outputs=cost, 
                                     updates=updates,
                                     allow_input_downcast=True)
        

    def __call__(self, UR, UL, LL, LR, learning_rate = 0.1, nb_iterations = 100):
        """
        Fits square on a set on points representing corners of the square. 
        Free parameters fitted are the square width, it's center position as 
        well as it's rotation angle.  Optimization is done using batch vanilla 
        gradient descent.  Initialization is done using average corner 
        positions.
        
        Parameters
        ----------
        UR, UL, LL, LR:     numpy.ndarray with shape [nb_examples, 2]
                            UR represents upper right corner examples, LL is 
                            for lower right corner examples, etc.
                            Each line of the array represents an example.
                            The first column represents x positions and the 
                            second column represents y positions

        learning_rate:      Decimal type
                            Gradient descent learning rate
        
        nb_iteration:       Integer type
                            Number of iterations of gradient descent
                            
        Returns
        -------
        position:           numpy.ndarray with shape [1,2]
                            Center position of the fitted square.
                            position[0,0] represents it's x position
                            position[0,1] represents it's y position
                            
        width:              float
                            Width of the fitted square
        
        theta:              float
                            Angle of the fittes square in radians
        """        
        
        # Set input data to shared variabled
        self.ur_data.set_value(UR)
        self.ul_data.set_value(UL)
        self.ll_data.set_value(LL)
        self.lr_data.set_value(LR)
        
        # Estimating center initial value
        self.p.set_value(((UR + UL + LL + LR).mean(0) / 4.)[None, :])
        
        # Estimating width initial value
        w = 0.25 * np.sqrt(((UR-UL)**2).sum(1)).mean() + \
            0.25 * np.sqrt(((UL-LL)**2).sum(1)).mean() + \
            0.25 * np.sqrt(((LL-LR)**2).sum(1)).mean() + \
            0.25 * np.sqrt(((LR-UR)**2).sum(1)).mean()
        self.w.set_value(w)
        
        # Estimating angle initial value
        edge   = (UR-UL).mean(0)
        sin_t  = edge[1] / np.sqrt((edge**2).sum())
        cos_t  = edge[0] / np.sqrt((edge**2).sum())
        
        edge   = (UL-LL).mean(0)
        sin_t -= edge[0] / np.sqrt((edge**2).sum())
        cos_t += edge[1] / np.sqrt((edge**2).sum())

        edge   = (LL-LR).mean(0)
        sin_t -= edge[1] / np.sqrt((edge**2).sum())
        cos_t -= edge[0] / np.sqrt((edge**2).sum())

        edge   = (LR-UR).mean(0)
        sin_t += edge[0] / np.sqrt((edge**2).sum())
        cos_t -= edge[1] / np.sqrt((edge**2).sum())
        
        self.t.set_value(atan2(sin_t/4., cos_t/4.))
        
        #p = self.p.get_value()
        #w = self.w.get_value()
        #t = self.t.get_value()
        #print "        Preliminary estimated to center (x,y) = ({:0.4f}, {:0.4f}), width = {:0.4f}, angle = {:0.4f} deg".format(p[0,0], p[0,1], w, t*180/np.pi)
        
        # Optimizing
        for i in range(nb_iterations):
            self.update_fn(learning_rate)
            
        return self.p.get_value(), self.w.get_value(), self.t.get_value()
            
square_fit_from_corners = _square_fit_from_corners()





class _rectangle_fit_from_corners(object):
    
    def __init__(self):
        
        self.learning_rate_t_fac = 1/10000000.
        
        # Convert corner data into shared variables   
        self.ur_data = th.shared(np.empty((0, 2), th.config.floatX), name = "ur_data", allow_downcast=True)
        self.ul_data = th.shared(np.empty((0, 2), th.config.floatX), name = "ul_data", allow_downcast=True)
        self.ll_data = th.shared(np.empty((0, 2), th.config.floatX), name = "ll_data", allow_downcast=True)
        self.lr_data = th.shared(np.empty((0, 2), th.config.floatX), name = "lr_data", allow_downcast=True)

        # Parameter of square to be fitted to the data
        self.p = th.shared(np.asarray([0,0], th.config.floatX)[None, :], 'center', allow_downcast=True)
        self.t = th.shared(np.asarray(0, th.config.floatX), 'theta', allow_downcast=True)
        self.w = th.shared(np.asarray(0, th.config.floatX), 'width', allow_downcast=True)
        self.h = th.shared(np.asarray(0, th.config.floatX), 'height', allow_downcast=True)
        params = [self.p, self.t, self.w, self.h]

        # Model corner data 
        self.ur_model = T.stacklists([ self.w/2,  self.h/2])
        self.ul_model = T.stacklists([-self.w/2,  self.h/2])
        self.ll_model = T.stacklists([-self.w/2, -self.h/2])
        self.lr_model = T.stacklists([ self.w/2, -self.h/2])
        
        # Rotation matrix associated to angle
        R = T.stacklists([[T.cos(self.t), -T.sin(self.t)],[T.sin(self.t), T.cos(self.t)]])
            
        # Symbolic cost function
        cost   = ((self.p + T.dot(self.ur_model, R.T) - self.ur_data)**2).sum()
        cost  += ((self.p + T.dot(self.ul_model, R.T) - self.ul_data)**2).sum()
        cost  += ((self.p + T.dot(self.ll_model, R.T) - self.ll_data)**2).sum()
        cost  += ((self.p + T.dot(self.lr_model, R.T) - self.lr_data)**2).sum()
        
        # Hyperparameters
        learning_rate = T.scalar('learning_rate', th.config.floatX)
        
        # Symbolic gradient
        grads = dd([(param, grad) for param, grad in zip(params, T.grad(cost=cost, wrt=params))])
        
        # Symbolic update
        updates = dd([(param, param - (learning_rate*self.learning_rate_t_fac if param==self.t else learning_rate)*grad) for param, grad in grads.items()])
        
        # Theano update function
        self.update_fn = th.function(inputs = [learning_rate], 
                                     outputs=cost, 
                                     updates=updates,
                                     allow_input_downcast=True)
        

    def __call__(self, UR, UL, LL, LR, learning_rate = 0.1, nb_iterations = 100):
        """
        Fits square on a set on points representing corners of the rectangle. 
        Free parameters fitted are the rectangle width and height, it's center 
        position as well as it's rotation angle.  Optimization is done using 
        batch vanilla gradient descent.  Initialization is done using average 
        corner positions.
        
        Parameters
        ----------
        UR, UL, LL, LR:     numpy.ndarray with shape [nb_examples, 2]
                            UR represents upper right corner examples, LL is 
                            for lower right corner examples, etc.
                            Each line of the array represents an example.
                            The first column represents x positions and the 
                            second column represents y positions

        learning_rate:      Decimal type
                            Gradient descent learning rate
        
        nb_iteration:       Integer type
                            Number of iterations of gradient descent
                            
        Returns
        -------
        position:           numpy.ndarray with shape [1,2]
                            Center position of the fitted square.
                            position[0,0] represents it's x position
                            position[0,1] represents it's y position
                            
        width:              float
                            Width of the fitted square

        width:              float
                            Height of the fitted square        

        theta:              float
                            Angle of the fitted square in radians
        """        
        
        # Set input data to shared variabled
        self.ur_data.set_value(UR)
        self.ul_data.set_value(UL)
        self.ll_data.set_value(LL)
        self.lr_data.set_value(LR)
        
        # Estimating center initial value
        self.p.set_value(((UR + UL + LL + LR).mean(0) / 4.)[None, :])
        
        # Estimating width initial value
        w = 0.25 * np.sqrt(((UR-UL)**2).sum(1)).mean() + \
            0.25 * np.sqrt(((LL-LR)**2).sum(1)).mean()
        self.w.set_value(np.asarray(w, dtype=th.config.floatX))

        # Estimating heightinitial value
        h = 0.25 * np.sqrt(((UL-LL)**2).sum(1)).mean() + \
            0.25 * np.sqrt(((LR-UR)**2).sum(1)).mean()
        self.h.set_value(np.asarray(h, dtype=th.config.floatX))

        
        # Estimating angle initial value
        edge   = (UR-UL).mean(0)
        sin_t  = edge[1] / np.sqrt((edge**2).sum())
        cos_t  = edge[0] / np.sqrt((edge**2).sum())
        
        edge   = (UL-LL).mean(0)
        sin_t -= edge[0] / np.sqrt((edge**2).sum())
        cos_t += edge[1] / np.sqrt((edge**2).sum())

        edge   = (LL-LR).mean(0)
        sin_t -= edge[1] / np.sqrt((edge**2).sum())
        cos_t -= edge[0] / np.sqrt((edge**2).sum())

        edge   = (LR-UR).mean(0)
        sin_t += edge[0] / np.sqrt((edge**2).sum())
        cos_t -= edge[1] / np.sqrt((edge**2).sum())
        
        self.t.set_value(np.asarray(atan2(sin_t/4., cos_t/4.), dtype=th.config.floatX))
        
        #p = self.p.get_value()
        #w = self.w.get_value()
        #t = self.t.get_value()
        #print "        Preliminary estimated to center (x,y) = ({:0.4f}, {:0.4f}), width = {:0.4f}, angle = {:0.4f} deg".format(p[0,0], p[0,1], w, t*180/np.pi)
        
        # Optimizing
        for i in range(nb_iterations):
            self.update_fn(learning_rate)
            
        return self.p.get_value(), self.w.get_value(), self.h.get_value(), self.t.get_value()
            
rectangle_fit_from_corners = _rectangle_fit_from_corners()



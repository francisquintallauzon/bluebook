# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 11:42:17 2015

@author: francis
"""

import numpy as np
import random
from libutils.dict import dd
from libutils.index import rolling_permutations as rp

class datamodel(object):
    def __init__(self):
        self.__name = self.__class__.__name__
        self.__examples = []
        self.__classes = []
        self.__splits = dd([('train', []), ('valid', []), ('test', []), ('full', [])])
        self.__rpg = dd() # random permutation generator
        self.splits.full = self.examples
        
    @property
    def classes(self):
        return self.__classes
        
    @classes.setter
    def classes(self, other):
        if id(other) != id(self.__classes):
            raise ValueError, "Property 'classes' cannot change id"        
        self.__classes = other
        
    @property
    def examples(self):
        return self.__examples
        
    @examples.setter
    def examples(self, other):
        if id(other) != id(self.__examples):
            raise ValueError, "Property 'examples' cannot change id"
        self.__examples = other
        
    @property
    def nb_examples(self, split_id = 'full'):
        return len(self.examples[split_id])
        
    @property
    def name(self):
        return self.__name
    
    @name.setter
    def name(self, other):
        self.__items = other
        
    @property
    def splits(self):
        return self.__splits


#    def get_random_examples(self, split_id, nb=1):
#        if split_id not in self.__rpg:
#            self.__rpg[split_id]= rp(len(self.splits[split_id]))
#
#        indices = self.__rpg[split_id](nb)
#        
#        if isinstance(indices, np.ndarray) :
#            return [self.splits[split_id][idx] for idx in indices]
#        else : 
#            return self.splits[split_id][indices]
        
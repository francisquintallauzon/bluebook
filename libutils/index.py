# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 21:49:09 2013

@author: francis
"""

import numpy as np

class rolling_permutations(object):
    def __init__(self, nb):
        self.nb = nb
        self.permutations = np.random.permutation(nb)
        self.current = 0

    def __call__(self, nb=1):
        i = 0
        ind = np.empty(nb, self.permutations.dtype)
        while(self.current + nb > self.nb):
            ind[i : i + self.nb - self.current] = self.permutations[self.current:]
            i += self.nb - self.current
            nb -= self.nb - self.current
            self.current = 0
            self.permutations = np.random.permutation(self.nb)
        ind[i : i + nb] = self.permutations[self.current:self.current+nb]
        self.current += nb
        return ind[0] if nb==1 else ind    
        
        
        
class rolling_indices(object):
    def __init__(self, nb):
        self.nb = nb
        self.current = 0

    def __call__(self, nb=1):
        i = 0
        ind = np.empty(nb, np.int)
        while(self.current + nb > self.nb):
            ind[i : i + self.nb - self.current] = np.arange(self.current, self.nb)
            i += self.nb - self.current
            nb -= self.nb - self.current
            self.current = 0
        ind[i : i + nb] = np.arange(self.current, self.current+nb)
        self.current += nb
        return ind    
        
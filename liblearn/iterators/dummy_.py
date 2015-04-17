# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 14:49:50 2015

@author: francis
"""

class dummy(object):

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration



# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 16:55:46 2014

@author: francis
"""


import inspect

def lineno():
    """
    Returns the current line number in the program.
    
    Examples
    -------- 
    To print the current line :
    
    >>> print lineno()
    19

    Note
    ----
    Code reproduced from Danny Yoo (dyoo@hkn.eecs.berkeley.edu) from
    http://code.activestate.com/recipes/145297-grabbing-the-current-line-number-easily/
    """

    return inspect.currentframe().f_back.f_lineno

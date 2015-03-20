# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 16:08:13 2015

@author: francis
"""

import cv2
import numpy as np
from os.path import join


class element(object):

    def __init__(self, path, fn, label=None, desc=None, origin = None, index = None):
        """
        This class contains all meta information necessary to represent a leukocyte example from the database
        as well as input function

        Parameters
        ----------
        path :      string
                    relative path to image folder

        fn :        string
                    element's filename

        label :     int
                    element's label

        desc :      str
                    element's label description

        origin :    string
                    dataset origin

        index :     any type
                    unique indentifier to object
        """
        self.path = path
        self.filename = fn

        if label:
            self.label = label

        if desc:
            self.desc = desc

        if origin :
            self.origin = origin

        if index :
            self.id = index

    def get_image(self):
        return cv2.imread(join(self.path, self.filename)).astype(np.float32)[:,:,::-1]/255.

    #def set_labelmap(self, labelmap):
    #    self.labelmap = labelmap

    #def get_mappedlabel(self):
    #    return self.labelmap[self.label]


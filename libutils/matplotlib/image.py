# -*- coding: utf-8 -*-
"""
Created on Wed Feb 06 16:34:17 2013

@author: francis
"""

import numpy as np

import matplotlib.pyplot as plt

def imshow(ax, img, title = None, xlabel = None, ylabel = None, cmap = 'gray', vmin = None, vmax = None, fontsize = 8):
    
    # Copy image
    img = img.copy()
    
    # Remove nans
    img[np.isnan(img)] = 0
    
    # Clear axes
    ax.cla()
    
    if img.ndim == 2:
        ax.imshow(img, vmin = vmin, vmax = vmax, interpolation='none').set_cmap(cmap)
    elif img.ndim == 3:
        ax.imshow(img, interpolation='none')

    if title is not None:
        ax.set_title(title, fontsize=fontsize)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize)

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize)
        
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize-2) 

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize-2) 


def debugimshow(img, title = None, xlabel = None, ylabel = None, cmap = 'gray', vmin = None, vmax = None, fontsize = 8):
    f = plt.figure()
    ax = f.add_subplot(1,1,1)
    imshow(ax,  img, title, xlabel, ylabel, cmap, vmin, vmax, fontsize)
    plt.show()

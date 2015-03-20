# -*- coding: utf-8 -*-
"""
Created on Wed Feb 06 16:34:07 2013

@author: francis
"""

import numbers
import exceptions
import numpy as np
import matplotlib.pyplot as plt

def hist(ax, data, threshold=None, pltx=None, plty=None, title = None, xlabel = None, ylabel = None, nbins = 256, vmin = None, vmax = None, fontsize = 8, clearaxes = True):

    # Clear axes
    if clearaxes :
        ax.cla()

    assert np.all(np.isnan(data)) == False, 'np.all(np.isnan(data)) = {}'.format(np.all(np.isnan(data)))
    n, bins, patches = ax.hist(data[~np.isnan(data)], bins = nbins, histtype = 'stepfilled')
    ax.set_yticklabels([])

    if threshold != None:
        if isinstance(threshold, np.ndarray):
            threshold = threshold.flatten().tolist()

        if isinstance(threshold, numbers.Number):
            threshold = [threshold]

        if not isinstance(threshold, list) :
            raise TypeError, 'type(threshold) = {} is not supported'.format(type(threshold))

        numel = len(threshold)

        colors = (plt.cm.get_cmap('jet', numel))(np.arange(numel))

        for c, t in zip(colors, threshold):
            #print 'printing threshold {} with color {}'.format(t, c)
            ax.axvline(t, 0, 1, color = c)

    if pltx != None and plty != None:
        ax.plot(pltx, plty, 'r--', linewidth=1)

    if title is not None:
        ax.set_title(title, fontsize=fontsize)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize)

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize)

    if not(vmin is None or vmax is None):
        ax.set_xlim(vmin, vmax)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize-2)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize-2)


def debughist(data, threshold = None, pltx=None, plty=None, title = None, xlabel = None, ylabel = None, nbins = 256, vmin = None, vmax = None, fontsize = 8):
    f = plt.figure()
    ax = f.add_subplot(1,1,1)
    hist(ax=ax,  data=data, threshold=threshold, pltx=pltx, plty=plty, title=title, xlabel=xlabel, ylabel=ylabel, nbins=nbins, vmin=vmin, vmax=vmax, fontsize=fontsize)
    plt.show()

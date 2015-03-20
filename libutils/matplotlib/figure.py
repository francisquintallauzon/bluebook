# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 20:41:08 2013

@author: francis
"""
if __name__ == '__main__':
    import sys
    sys.path.append("../../")


import numpy                        as np
import matplotlib.gridspec          as gridspec
from libutils.function        import timing
from matplotlib.figure              import Figure
from matplotlib.axes                import Axes


def subplots(nb_lin, nb_col, height=None, width=None, projection = None, **figure_properties):
    from matplotlib.pyplot import figure
    height = height if height else nb_lin * 3
    width = width if width else nb_col * 3
    return figure(FigureClass=subplots_, nb_lin=nb_lin, nb_col=nb_col, height=height, width=width, projection=projection, **figure_properties)


class subplots_(Figure):

    def __init__(self, nb_lin, nb_col, height, width, projection = None, **figure_properties):
        """
        Prepare a figure with a specified number axes

        Parameters
        ----------
        nb_lin, nb_lin      integer
                            Number of lines and number of columns

        height, width       float
                            Height and width of the figure in inches

        axes_properties:    set of axes properties
        """

        figure_properties['figsize'] = (width, height)

        Figure.__init__(self, **figure_properties)

        self.nb_lin = nb_lin
        self.nb_col = nb_col

        gs = gridspec.GridSpec(nb_lin, nb_col)
        self.ax = np.ndarray((nb_lin, nb_col), dtype = Axes)
        for row in range(gs._nrows):
            for col in range(gs._ncols):
                self.ax[row,col] = self.add_subplot(gs[row,col], projection=projection)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self.ax[key]
        else :
            return self.ax[0,key]


    @timing
    def save(self, fn, dpi=300, bbox_inches='tight', pad_inches=0.2, *args, **kwargs):
        """
        Save figure to file.  Simple wrapper for the savefig method
        """

        self.savefig(fn, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches, *args, **kwargs)


if __name__ == '__main__':
    sp = subplots(1,1,1,1)
    sp = subplots(1,1,1,1, projection='recurrent')

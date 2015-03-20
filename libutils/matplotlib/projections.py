# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 14:49:25 2014

@author: francis
"""
if __name__ == '__main__':
    import sys
    sys.path.append("../../")


import numpy as np
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
from matplotlib.projections import register_projection
from libutils.dict import dd
from libutils.function import timing


class axes_base(Axes):

    name = 'improved'

    def __init__(self, *args, **kwargs):
        self.__patches_hist = []
        Axes.__init__(self, *args, **kwargs)


    def hist(self, remove=False, *args, **kwargs):

        # Remove old patches from an old histogram
        if remove:
            for i in range(len(self.patches)-1, -1, -1):
                del self.patches[i]
            kwargs['color'] = matplotlib.rcParams['axes.color_cycle'][0]

        x = np.asarray(kwargs['x'] if 'x' in kwargs else args[0]).flatten()

        # Remove nans
        x = x[~np.isnan(x)]

        if 'x' in kwargs:
            kwargs['x'] = x
        else:
            args[0] = x

        patches = None
        if x.size > 1:
            patches = Axes.hist(self, *args, **kwargs)
            self.relim()
            self.autoscale()

        return patches




    def set(self, xtick_fontsize=None, ytick_fontsize=None, **kwargs):
        if xtick_fontsize:
            for tick in self.xaxis.get_major_ticks():
                tick.label.set_fontsize(xtick_fontsize)

        if ytick_fontsize:
            for tick in self.yaxis.get_major_ticks():
                tick.label.set_fontsize(ytick_fontsize)

        Axes.set(self, **kwargs)



class axes_recurrent(axes_base):

    name = 'recurrent'

    def __init__(self, *args, **kwargs):
        axes_base.__init__(self, *args, **kwargs)
        self.__lines = {}
        self.__colorcycle = matplotlib.rcParams['axes.color_cycle']
        self.__colorcycle_id = 0

    @timing
    def add_line(self, line=None, name=None, x=None, y=None, **line_kwargs):

        if line is not None and name is None:
            if line.get_label():
                name = line.get_label()

        if line is None and name is None:
            if 'label' in line_kwargs:
                name = line_kwargs['label']

        if name in self.__lines:
            raise ValueError, "A line referenced by name {} already exists".format(name)

        x = x if isinstance(x, list) else []
        y = y if isinstance(y, list) else []

        if not line :
            if 'color' not in line_kwargs:
                line_kwargs['color'] = self.__colorcycle[self.__colorcycle_id % len(self.__colorcycle)]
                self.__colorcycle_id += 1
            line = Line2D(x, y, **line_kwargs)

        if name :
            line_dd = dd()
            line_dd.x = x
            line_dd.y = y
            line_dd.line = line
            self.__lines[name] = line_dd

        axes_base.add_line(self, line)
        self.relim()
        self.autoscale()


    @timing
    def add_point(self, **points):
        x_offset = 1 if self.get_xscale()=='log' else 0
        for name, val in points.items():
            d = self.__lines[name]
            x, y = val if isinstance(val, tuple) else (len(d.x) + x_offset, val)
            d.x += [x]
            d.y += [y]
            d.line.set_xdata(np.asarray(d.x))
            d.line.set_ydata(np.asarray(d.y))
        self.relim()
        try :
            self.autoscale()
        except :
            pass


register_projection(axes_base)
register_projection(axes_recurrent)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111, projection='recurrent')
    ax.add_line(name = 'line', x = [0.1,1,2,3,4], y=[0,1,2,2,1])
    ax.add_point(line=(10.5,1))
    plt.show()


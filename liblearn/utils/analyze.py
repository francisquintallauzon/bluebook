# -*- coding: utf-8 -*-
"""
Created on Fri May  3 17:05:33 2013

@author: root
"""
# Library imports
import os
import numpy                        as np
import matplotlib.pyplot            as plt

# Local imports
from libutils.path            import walk_folders
from libutils.path            import make_dir
from libutils.path            import delete
from libutils.dict            import dd
from libutils.string          import sstr
from traceback                import print_exc
from math                     import log



def analyze(experiment_path, output_path, objectives, ignores, ignore_vals, substitute, out_to_hp):

    # Remove unfinished experiments
    clean(experiment_path, objectives)

    # Get experiment dictionary
    lpath = walk_folders(experiment_path)

    # Get all experiment files
    ehp = expdd()
    eout = expdd()
    for path in lpath:
        try:
            hp = read_hp(os.path.join(path, 'hp.txt'), substitute)
            out = read_hp(os.path.join(path, 'out.txt'), substitute)
        except:
            continue

        ehp += hp
        eout += out

    # Make output directory
    make_dir(output_path)

    # Transfer output data to input
    for obj in out_to_hp:
        if obj in eout:
            ehp["out."+obj] = dd()
            ehp["out."+obj] = eout[obj]

    # Get exclusion
    arr, _ = ehp.get_array(list(ehp.keys())[0])
    include = np.ones(arr.size)
    if ignore_vals != None:
        for key, value in list(ignore_vals.items()):
            arr, _ = ehp.get_array(key)
            include *= (arr != value)

    fig = plt.figure()  #(figsize = (4,3))

    for xlabel in list(ehp.keys()):

        do_ignore = False
        for ignore in ignores:
            print(ignore, xlabel, ignore in xlabel)
            if ignore in xlabel:
                do_ignore = True
                break

        if do_ignore:
            continue

        fig.clf()
        ax = fig.add_subplot(111)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Objective')
        ax.set_title('Hyperparameter optimization for {}'.format(os.path.split(experiment_path)[1]))
        x, this_include = ehp.get_array(xlabel)
        this_include *= include

        # Do not output hyperparameters with fixed size
        if np.unique(x).size == 1:
            continue

        print("For {}, number of unique x is {}".format(xlabel, np.unique(x).size))
        if np.unique(x).size == 2:
            for val in np.unique(x):
                print('        {}'.format(val))

        # Convert object dtype to string
        if x.dtype == np.object:
            x = [str(obj) for obj in x]

        # Rotate string to 90 degrees for readability of graph
        if isinstance(x[0], str):

            for i in range(len(x)):
                x[i] = x[i].replace("\\",  "/")

                if 'path' in xlabel:
                    tail, head = os.path.split(x[i])
                    x[i] = os.path.join(os.path.split(tail)[1], head)

            un = np.unique(x).astype(sstr)
            xticks = un.copy()
            un = dict(list(zip(un, np.arange(un.size))))
            x = np.asarray([un[val] for val in x], dtype = sstr)
            ax.set_xticks(np.arange(xticks.size))
            ax.set_xticklabels(xticks)

            for ticklabel in ax.get_xticklabels():
                ticklabel.set_rotation(90)
                if 'path' in xlabel:
                    ticklabel.set_fontsize(3)

        for ylabel in list(eout.keys()):
            if ylabel in objectives:
                y, _ = eout.get_array(ylabel)
                ax.plot(x[this_include * (y!=0)],y[this_include * (y!=0)], marker = '.', markersize = 5, linestyle='none', label = ylabel)

        try:
            xlim = ax.get_xlim()
            ax.set_xlim(xlim[0]-(xlim[1]-xlim[0])*0.05, xlim[1]+(xlim[1]-xlim[0])*0.05)
            ax.legend(loc=1, fontsize=7)
            #ax.set_yscale('log')
            fig.savefig(os.path.join(output_path, os.path.split(experiment_path)[1] + "." + xlabel + '.png'), bbox_inches='tight', dpi=600)
        except:
            print_exc()

def clean(experiment_path, objectives):

    paths = walk_folders(experiment_path)

    for path in paths:

        do_delete = False

        if os.path.isfile(os.path.join(path, 'hp.txt')) and not os.path.isfile(os.path.join(path, 'out.txt')):
            do_delete = True

        if do_delete:
            print("deleting {} because experiment was incomplete".format(path))
            delete(path)


def read_hp(fn, substitute=None):
    hp = {}
    with open(fn) as f:
        for line in f:
            key, val = line.partition("=")[::2]

            # Substitute
            if substitute != None:
                if key in substitute:
                    key = substitute[key]

            val = val.replace('\n','')
            if val[-1] == chr(13):
                val = val[:-1]


            if val == 'True':
                hp[key] = True
            elif val == 'False':
                hp[key] = False
            else:
                try:
                    hp[key] = int(val)
                except:
                    try:
                        hp[key] = float(val)
                    except:
                        if "[" in val and "]" in val:
                            val = val.replace(" ", ", ")
                            val = val.replace("[", "(")
                            val = val.replace("]", ")")
                        hp[key] = sstr(val)
    return hp



class expdd(dict):


    def __init__(self):
        # Current number of experiment results
        self._nb = 0


    def get_array(self, key):

        ####################################################################
        # 1. infer type key using votation scheme

        # Get votes
        typed = {}
        for val in self[key].value:
            if type(val) in typed:
                typed[type(val)] += 1
            else :
                typed[type(val)] = 1

        # Count votes
        max_votes = 0
        for key_t, val in list(typed.items()):
            if val > max_votes:
                winner_t = key_t
                max_votes = val

        ####################################################################
        # 2. Format output array

        output = np.zeros(self._nb, dtype=winner_t)
        try :
            output[self[key].index] = self[key].value
            include = np.zeros(output.size, dtype=np.bool)
            include[self[key].index]=True
        except :
            print("key = ", key)
            print("output.shape = ", output.shape)
            print("self[key].index = ", self[key].index)
            print("self[key].value = ", self[key].value)
            print("winner_t = {}; max_votes = {}".format(winner_t, max_votes))
            raise

        return output, include


    def __iadd__(self, other):

        if not isinstance(other, dict):
            raise TypeError("__iadd__ rhs is of must be or derive from type dict.  Rather, is of type type {}".format(type(other)))

        self._nb += 1

        for key, value in list(other.items()):

            if key in self:
                self[key].value += [value]
                self[key].index += [self._nb-1]
            else:
                self[key] = dd()
                self[key].value = [value]
                self[key].index = [self._nb-1]

        return self



# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 15:58:43 2015

@author: francis
"""

from libutils.dict        import dd
from liblearn.datamodel   import datamodel

class marshall(datamodel):

    def __init__(self, classmap_path=None, **models):
        super(marshall, self).__init__()

        self.models = models.values()

        # Classmap maps class name from the dataset domain to generalized class names
        self.classmap = self.__loadmapping(classmap_path)

        # Extract classnames
        self.classes += [c.desc for c in self.classmap.values()]

        # Marchall all classes and convert class description
        frequency = dd([(cls, 0.) for cls in self.classes])
        for model in self.models:
            for e in model.examples:
                e.label = self.classmap[e.desc].label
                e.desc = self.classmap[e.desc].desc
                self.examples += [e]
                frequency[e.desc] += 1

        # Normalize frequencies
        for cls in frequency:
            frequency[cls] /= len(self.examples)

        # Report number of examples per class
        print "Marshalling datasets with a total of {} examples".format(self.nb_examples)
        for cls, freq in frequency.items():
            print "    {:7d} ({:6.3f}%) items of class {}".format(int(freq*self.nb_examples), freq*100., cls)


    def __loadmapping(self, fn):
        classmap = dd()
        labelmap = dd()
        currentlabel = -1
        with open(fn, 'rb') as f:
            for line in f:
                
                # Split line
                line = line.replace('\n', '')
                split = line.split('=')
                
                # Extract key and value
                key = split[0]
                val = split[len(split) == 2]
                
                # Add class description if not in label mapping (from class description to class number)
                if val not in labelmap:
                    currentlabel += 1
                    labelmap[val] = currentlabel
                
                # Add key to classmap 
                if key not in classmap :
                    classmap[key] = dd()
                    
                # Assign description and label to key
                classmap[key].label = labelmap[val]
                classmap[key].desc  = val

        return classmap


if __name__ == '__main__':
    import datasets
    from os.path import join
    from cellavision_ import cellavision
    from clemex_ import clemex

    clmx = clemex(join(datasets.path(), 'leuko', 'clemex'))
    clvs = cellavision(join(datasets.path(), 'leuko', 'cellavision', 'images'))

    classmap_path = join(datasets.path(), 'leuko', 'classmap.txt')
    marshall(classmap_path, s1=clmx, s2=clvs)
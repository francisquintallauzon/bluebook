# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 15:58:43 2015

@author: francis
"""

from os.path import join
from libutils.path import walk_files
from libutils.dict import dd
from liblearn.datamodel import element, datamodel


class source(datamodel):

    # Regular expression for finding class from filename

    def __init__(self, path=None):
        datamodel.__init__(self)

        self.path = path if path else self.path

        self.splits = dd({split:[] for split in ['train', 'valid', 'test']})

        # Maps class label to a list of images for each given class
        for fn in walk_files(self.path, join=False, explore_subfolders=False):

            # example class name
            cls = fn[:3]
            idx = int(fn[4:-4])

            # Add class to class list
            if cls not in self.classes:
                self.classes.append(cls)

            example = element(self.path, fn, label=self.classes.index(cls), desc=cls, index=idx)

            self.examples.append(example)

            if idx < 20000//2:
                self.splits.train.append(example)
            elif idx < 22500//2:
                self.splits.valid.append(example)
            else :
                self.splits.test.append(example)

        print("Cats and dogs dataset is loaded with:")
        for split_id in self.splits:
            print("    {:7d} examples in {} set".format(len(self.splits[split_id]), split_id))


if __name__ == '__main__':
    import datasets
    s = source(join(datasets.path(), 'catsanddogs'))
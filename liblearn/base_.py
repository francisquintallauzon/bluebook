# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 15:37:50 2014

@author: francis
"""

if __name__ == '__main__':
    import sys
    sys.path.append("../../")


import numpy           as np
from os.path           import join
from libutils.path     import make_dir
from libutils.dict     import dd
from libutils.function import classproperty
from liblearn.utils    import shared_x


class base(object):

    def __init__(self, params=None):
        self._params = dd() if params is None else params
        self.__W = None

    @classproperty
    @classmethod
    def name(cls):
        return  cls.__name__

    def __str__(self):
        return self.name

    def __call__(self, *args, **kwargs):
        """
        Must be implemented by derived class where this function's arguments
        must minimally contain the input to the layer and this function should
        return the layer's output
        """
        raise NotImplementedError


	def __getattribute__(self, key):
         in_attributes = key in object.__getattribute__(self, '__dict__')
         in_params = key in object.__getattribute__(self, '_params')

         if in_attributes and in_params:
             raise RuntimeError("key {} cannot be both in self.__dict__ and self._params".format(key))

         return object.__getattribute__(self, key)


    def __getattr__(self, key):
        params = object.__getattribute__(self, '_params')
        for name in params:
            if key is name:
                return params[key].value


    def __setattr__(self, key, val):

        try:
            in_attributes = key in object.__getattribute__(self, '__dict__')
            in_params = key in object.__getattribute__(self, '_params')

            if in_attributes and in_params:
                raise RuntimeError("key {} cannot be both in self.__dict__ and self._params".format(key))

            if not in_attributes and in_params:
                object.__getattribute__(self, '_params')[key].value = val
                return

        except:
            pass

        super(base, self).__setattr__(key, val)


    @property
    def optimizables(self):
        return [p.value for p in list(self._params.values()) if p.is_optimizable]

    @property
    def params(self):
        return [p.value for p in list(self._params.values()) if not p.is_hyperparam]

    @property
    def hparams(self):
        return [p.value for p in list(self._params.values()) if not p.is_hyperparam]



    def add_param(self, optimizable, **param):
        if len(param) > 1:
            raise ValueError('Call to add_param can only have one input param.  **param = {}'.format(param))

        name, value = list(param.items())[0]

        new = dd()
        new.value = value
        new.is_optimizable = optimizable
        new.is_hyperparam = False
        self._params[name] = new


    def add_hparam(self, **hparam):

        if len(hparam) > 1:
            raise ValueError('Call to add_param can only have one hparam.  **hparam = {}'.format(hparam))

        name, value = list(hparam.items())[0]

        new = dd()
        new.value = value
        new.is_optimizable = False
        new.is_hyperparam = True
        self._params[name] = new


    def shape(self, input_sz):
        """
        Layer output shape given input size speficied by input_sz
        """
        return input_sz


    def export(self, export_path):
        """
        Export layer parameters to the specified export path
        """

        # Make output directory
        if self._params or self.hparams:
            make_dir(export_path)

        # Save parameter files to npy format.  Set filename to value field of parameters
        for key, param in list(self._params.items()):
            if not param.is_hyperparam:
                np.save(join(export_path, "{}.npy".format(key)), param.value.get_value())
                param.name = param.value.name
                param.value = "{}.npy".format(key)

        # Pickle hyperparameters
        self._params.dump(join(export_path, "params.pkl"))


    @classmethod
    def load(cls, import_path):

        # Load hyperparameter dict
        params = dd.load(join(import_path, 'params.pkl'))

        # Load parameter files
        for key, param in list(params.items()):
            if not param.is_hyperparam:
                param.value = shared_x(np.load(join(import_path, param.value)), name=param.name)
                del param['name']

        # Initialize class
        return cls(params)


if __name__ == '__main__':

    test = base()
    print('addparam')
    test.add_param(param = shared_x(np.zeros((30,30)), name = 'p'), optimizable = True)
    print('addhparam')
    test.add_hparam(hparam = 1)
    print('export')
    test.export('.')
    print('load')
    rtest = base.load('.')
    print(rtest.params)

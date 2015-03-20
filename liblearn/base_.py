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

    def __init__(self, hparams, params={}):
        self.params = dd(params)
        self.hparams = dd(hparams)
        
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
         in_params = key in object.__getattribute__(self, 'params')
         in_hparams = key in object.__getattribute__(self, 'hparams')
          
         if (in_attributes and in_params) or (in_attributes and in_hparams):
             raise RuntimeError, "key {} cannot be both in self.__dict__ and self.params or self.hparams".format(key)
         return object.__getattribute__(self, key)


    def __getattr__(self, key):
        try : 
            return object.__getattribute__(self, 'params')[key]
        except:
            try:
                return object.__getattribute__(self, 'hparams')[key]
            except:
                raise


    def __setattr__(self, key, val):

        if key not in self.__dict__:
 
            try:
                if key in self.params:
                    self.params[key] = val
                    return
            except: 
                pass

            try :
                if key in self.hparams:
                    self.hparams[key] = val
                    return
            except: 
                pass

        super(base, self).__setattr__(key, val)


    def learn(self):
        pass


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
        if self.params or self.hparams:
            make_dir(export_path)

        # Dump hyperparameter dict
        self.hparams.dump(join(export_path, 'hparams.pkl'))
        
        # Dump dict mapping parameter name to parameter file name
        pmap = dd({key: "{}.npy".format(key) for key in self.params})
        pmap.dump(join(export_path, 'params.pkl'))
        
        # Save parameter files
        for key in self.params:
            np.save(join(export_path, pmap[key]), self.params[key].get_value())
            
    
    @classmethod
    def load(cls, import_path):
        
        # Load hyperparameter dict
        hparams = dd.load(join(import_path, 'hparams.pkl'))
        
        # Load the dict mapping parameter name to parameter file name
        params = dd.load(join(import_path, 'params.pkl'))
        
        # Load parameter files
        for key in params:
            params[key] = shared_x(np.load(join(import_path, params[key])), name=key)
            
        # Initialize class
        return cls( **(params + hparams) )

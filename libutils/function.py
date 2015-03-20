# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 14:50:36 2014

@author: francis
"""

from weakref import WeakKeyDictionary

class basicdescriptor(object):
    """A basic descriptor function"""
    def __init__(self):
        self.data = WeakKeyDictionary()

    def __get__(self, instance, owner):
        return self.data.get(instance)

    def __set__(self, instance, value):
        self.data[instance] = value


class propertydescriptor(basicdescriptor):
    """
    A descriptor that returns the instance that was set.  In the case where the
    instance is a function, it calls the function and returns it's output so
    that descriptor has a property-like behavior
    """
    def __get__(self, instance, owner):
        'call to propertydescriptor.__get__ from {}'.format(instance)
        data = self.data.get(instance)
        if hasattr(data, "__call__"):
            return data()
        else:
            return data


def timing(input_fn, *args, **kwargs):
    """
    The timing decorator adds a elapsed method to the decorated input_fn.  
    It that computes it's execution time each time input_fn is called. 
    
    input_fn.elapsed will return input_fn's execution time the last time it was called.
    
    This decorator is NOT thread safe
    """
    from time import time
    if hasattr(input_fn, "elapsed"):
        raise AttributeError, "Cannot decorate input_fn because it already has and 'elapsed' attribute"
    def new(*args, **kwargs):
        st = time()
        ret = input_fn(*args, **kwargs)
        new.elapsed = time()-st
        return ret
    new.__dict__ = input_fn.__dict__
    new.elapsed = 0
    return new

#    class new(object):
#        def __init__(self):
#            self.__elapsed = 0
#        
#        def __call__(self, *args, **kwargs):
#            st = time()
#            retval = input_fn(*args, **kwargs)
#            self.__elapsed = time()-st
#            return retval
#        
#        @property
#        def elapsed(self):
#            e = self.__elapsed
#            self.__elapsed = 0
#            return e
#    
#    return new()



class classproperty(property):
    """
    The exception decorator adds an exception method to the decorated input_fn
    that catches any exception raised by the function.  If an exception is
    caught, it the output from sys.exc_info() is stored in a .exception member.
    If not exception is caught, input_fn.exception will be None.
    """
    def __get__(self, cls, owner):
        return self.fget.__get__(None, owner)()


def exception(input_fn, *args, **kwargs):
    """
    The exception decorator adds an exception method to the decorated input_fn
    that catches any exception raised by the function.  If an exception is
    caught, it the output from sys.exc_info() is stored in a .exception member.
    If not exception is caught, input_fn.exception will be None.

    Call input_fn.exception after a calling input_fn() to get the last's call's
    exception.
    """

    if hasattr(input_fn, "exception"):
        raise AttributeError, "Cannot decorate input_fn because it already has and 'exception' attribute"
    def new(*args, **kwargs):
        from sys import exc_info
        try :
            new.exception = None
            ret = input_fn(*args, **kwargs)
        except:
            new.exception = exc_info()
            raise
        return ret
    new.__dict__ = input_fn.__dict__
    new.exception = None
    return new


def noraise(input_fn):
    """
    Try-except-pass
    """

    def new(*args, **kwargs):
        try :
            return input_fn(*args, **kwargs)
        except:
            pass
    new.__dict__ = input_fn.__dict__
    return new


if __name__ == '__main__':
    from time import sleep

    @timing
    @exception
    def func_1():
        sleep(1)
        
    func_1()
    print func_1.elapsed
    print func_1.elapsed


    
        
    

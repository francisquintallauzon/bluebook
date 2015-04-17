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


def timing(input_fn):
    from time import time
    from types import MethodType
    """
    The timing decorator adds a elapsed, cound and average methods to the decorated input_fn.

    input_fn.elapsed will return input_fn's total execution time
    input_fn.count will return input_fn's total number of executions
    input_fn.average will return input_fn's average execution time
    """
    class override(object):

        def __init__(self, function):
            self.__function = function
            self.__elapsed = 0
            self.__count = 0
            self.__last = 0

        def __call__(self, *args, **kwargs):
            self.__count += 1
            st = time()
            ret = self.__function(*args, **kwargs)
            self.__last = time()-st
            self.__elapsed += self.__last
            return ret

        def __get__(self, instance, owner):
            return MethodType(self, instance, owner)

        @property
        def elapsed(self):
            return self.__elapsed

        @property
        def count(self):
            return self.__count

        @property
        def average(self):
            return self.__elapsed / self.__count

        @property
        def last(self):
            return self.__last

    retval = override(input_fn)
    return retval


class addstatic(object):

    def __init__(self, **statics):
        self.statics = statics

    def __call__(self, function):

        class override(object):

            def __init__(self, function, statics):
                from libutils.dict import dd
                self.__function = function
                self.statics = dd(statics)

            def __call__(self, *args, **kwargs):
                return self.__function(*args, **kwargs)

            def __get__(self, instance, objtype):
                from types import MethodType
                return MethodType(self, instance)

        retval = override(function, self.statics)
        return retval


class classproperty(property):
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
        raise AttributeError("Cannot decorate input_fn because it already has and 'exception' attribute")
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

    class cls(object):

        @timer
        def fnc(self):
            sleep(1)

    c = cls()
    c.fnc()
    c.fnc()

    print(c.fnc.elapsed)
    print(c.fnc.count)
    print(c.fnc.average)






# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 14:56:56 2014

@author: francis
"""

import os
import cv2
import numpy                as np
from datetime               import datetime
from PIL                    import Image
from utils.path             import make_dir
from utils.path             import delete
from subprocess             import Popen as popen

def read_paged_tiff(fn):

    """
    Reads a paged tiff file using the PIL library

    Parameters
    ----------
    fn :        string
                path to tif file

    Returns
    -------
    img_list    list of numpy arrays
                list of images where element i of the list corresponds to page i of the tiff file
    """

    pil_image = Image.open(fn)

    img_list = []

    i = 0
    while(True):
        try:
            pil_image.seek(i)
            img_list += [np.array(pil_image)]
            i += 1
        except:
            break

    return img_list


def write_paged_tiff(fn, img_list):

    """
    Reads a paged tiff file using the ImageMagick library.
    This is really ugly but didn't find any other way...

    Parameters
    ----------
    fn :        string
                path to tif file

    img_list    list of numpy arrays
                list of images where element i of the list corresponds to page i of the tiff file
    """

    path = os.path.split(fn)[0]
    for i in range(1000):
        time = datetime.now().strftime("%Y%m%d_%Hh%Mm%Ss_%f")
        stamp = str(np.random.randint(10000000, 99999999))
        tmp = os.path.join(path, time+stamp)
        if not (os.path.isdir(tmp)):
            make_dir(tmp)
            try:
                tmp_fn_list = []
                for k, img in enumerate(img_list):
                    if img.ndim == 2:
                        write_img = (img * 255.0).astype(np.uint8)
                    elif img.ndim == 3:
                        write_img = (img[:,:,::-1] * 255.0).astype(np.uint8)
                    tmp_fn = os.path.join(tmp, '{}.tif'.format(k))
                    tmp_fn_list += [tmp_fn]
                    cv2.imwrite(tmp_fn, write_img)

                call = 'convert'
                for img_fn in tmp_fn_list:
                    call += ' '
                    call += img_fn
                call += ' '
                call += fn
                p = popen(call, env = os.environ, shell=True)
                p.wait()
            except:
                delete(tmp)
                raise
            delete(tmp)
            return

    raise IOError, 'Could not save {} because a hidden temporary directory could not be created'.format(fn)



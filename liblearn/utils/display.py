# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 15:09:28 2013

@author: root
"""

if __name__ == '__main__':
    import sys
    sys.path.append("../../")

from math import *
import cv2
import numpy as np
import theano as th

def imsave(img, fn):

    if img.ndim != 2 and img.ndim != 3:
        raise NotImplementedError('Expect img.ndim (={}) to be 2 or 3'.format(img.ndim))

    if img.ndim == 2:
        img = (img * 255.0).astype(np.uint8)

    if img.ndim == 3:
        img = img[:,:,::-1]

    if not np.issubdtype(img.dtype, np.integer):
        img *= 255.0

    if not np.issubdtype(img.dtype, np.uint8):
        img = img.astype(np.uint8)

    cv2.imwrite(fn, img)


def filterstoimg(W, size, layout=None, scale = 'local', fn = None):
    """
    Display a whole stack (colunmwise) of vectorized matrices. Useful
    to display the weights of a neural network layer.

    Parameters
    ----------
    W :         Numpy array of size [nb_pixels x nb_filters]
                Each of the N column represents a filter with size

    size:       tuple of len 2 or 3
                Size of input images.  W.ndims == 4, then size is ignored
                if len(size) == 2, then height, width = size (assumes grayscale image)
                if len(size) == 3, then height, width, channel = size

    layout      tuple of 2
                size of displayed image in number of patches

    scale       string
                if 'local', then the filter is scaled with respect to itself
                if 'global', then the filter is scaled with respect to the min and max of the whole W matrix

    fn          string
                output file of the resulting vizualization

    """

    # Border width
    bw = 2

    # Extract size information
    if len(size) == 3:
        height, width, nb_channels = size
    elif len(size) == 2:
        height, width = size
        nb_channels = 1
    else:
        raise ValueError("Dimension of input size (={}) should be 2 or 3".format(len(size)))

    nb_filters = W.shape[1]

    if layout is None:
        nlin = int(ceil(sqrt(nb_filters)))
        ncol = int(ceil(sqrt(nb_filters)))
    else:
        nlin, ncol = layout

    img = np.zeros(((height+bw)*nlin+bw, (width+bw)*ncol+bw, nb_channels), dtype = th.config.floatX)

    if scale == 'global':
        pmin = W.min()
        pmax = W.max()

    for k in range(nb_filters):
        # Tile position
        tcol = int(k % nlin)
        tlin = int(floor(k/nlin))

        #Weights to patch
        patch = W[:,k].reshape((height, width, nb_channels))
        if scale == 'global':
            patch = (patch - pmin) / (pmax - pmin)
        else :
            patch = (patch - patch.min()) / (patch.max() - patch.min())

        # Fill image
        img[tcol*(height+bw)+bw:(tcol+1)*(height+bw), tlin*(width+bw)+bw:(tlin+1)*(width+bw)] = patch


    # Remove this axis if singleton
    if img.shape[2] == 1:
        img = img.squeeze(2)

    if fn != None:
        imsave(img, fn)


    return img


def convfilterstoimg(W, layout=None, scale = 'local', fn = None):
    """
    Display convolutional filters.

    Parameters
    ----------
    W :         Numpy array of size [nb_filters x nb_channels x nb_lin x nb_col]
                Each slice W[i :,:,:] represents an image

    layout      tuple of 2
                size of displayed image in number of patches

    scale       string
                if 'local', then the filter is scaled with respect to itself
                if 'global', then the filter is scaled with respect to the min and max of the whole W matrix

    fn          string
                output file of the resulting vizualization
    """

    # Border width
    bw = 2

    # Extract size information
    if W.ndim != 4:
        raise ValueError("Number of dimension of W (={}) should be 4".format(W.ndim))

    if not(W.shape[1] == 1 or W.shape[1] == 3 or W.shape[1] == 4):
        raise ValueError("Shape of second dimension (i.e. nb_channels) of W (= {}), but should be 1, 3 or 4".format(W.shape[1]))

    nb_channels, height, width = W.shape[1:]

    nb_filters = W.shape[0]

    # print nb_filters, nb_channels, height, width

    if layout is None:
        nlin = int(ceil(sqrt(nb_filters)))
        ncol = int(ceil(sqrt(nb_filters)))
    else:
        nlin, ncol = layout

    img = np.zeros(((height+bw)*nlin+bw, (width+bw)*ncol+bw, nb_channels), dtype = th.config.floatX)

    if scale == 'global':
        pmin = W.min()
        pmax = W.max()

    for k in range(nb_filters):
        # Tile position
        tcol = int(k % nlin)
        tlin = int(floor(k/nlin))

        #Weights to patch
        patch = W[k].transpose(1,2,0)
        if scale == 'global':
            patch = (patch - pmin) / (pmax - pmin)
        else :
            patch = (patch - patch.min()) / (patch.max() - patch.min())

        # Fill image
        img[tcol*(height+bw)+bw:(tcol+1)*(height+bw), tlin*(width+bw)+bw:(tlin+1)*(width+bw)] = patch

    # Remove third dimension if singleton
    if img.shape[2] == 1:
        img = img.squeeze(2)

    # Save image if filename is provided
    if fn != None:
        imsave(img, fn)

    return img


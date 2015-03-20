# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:34:30 2013

@author: root
"""
import numpy                                as np
from math                                   import floor
from datetime                               import datetime
from sklearn.feature_extraction.image       import extract_patches_2d

def random_patches(image, patch_sz, nb_patches=1):
    return extract_patches_2d(image, patch_sz, nb_patches, datetime.now().microsecond)

def random_patches_with_replacement(image, labels, image_patch_sz, label_patch_sz, nb_patches, prob=None):
    """
    Extract patches at random locations along with its corresponding labels.

    An optional prob parameter can be passed so that label i be chosen with a probability of balance[i].
    This parameter was added in order to tackle the inter-class imbalance problem.  See the following paper for details:

    An optional mask can be used, so that only pixels for which mask is nonzero are considered.  Mind that this function becomes
    considerabely slow as the number of zeros in mask increases.

    Haibo He and Edwardo A. Garcia. 2009. Learning from Imbalanced Data. IEEE Trans. on Knowl. and Data Eng. 21, 9
    (September 2009), 1263-1284. DOI=10.1109/TKDE.2008.239 http://dx.doi.org/10.1109/TKDE.2008.239

    Parameters
    ----------
    image :         numpy.ndarray
                    [MxNxC] image from which to extract patches where M is the number of lines, N is the number of columns
                    and C is the number of channels

    labels :        numpy.ndarray
                    [MxN] label image from which to extract patches

    image_patch_sz: tuple of two integers
                    The size of patches to be extracted on the image

    label_patch_sz: tuple of two integers
                    The size of patches to be extracted on the label image

    nb_patches:     integer
                    number of patches to be extracted

    prob:           [optional] numpy.ndarray
                    Probability of selection for each class, such that the probability to select label[lin,col]==k is balance[k]
                    Therefore, if there is 4 possible labels then balance.shape == (number_of_labels,)

    Returns
    -------
    patches_image   numpy.ndarray
                    A [nb_patches x C X M x N] image

    patches_label   numpy.ndarray
                    A [nb_patches x 1 X M x N] image

    """

    # Initializations

    nb_channels = 1 if image.ndim==2 else image.shape[2]
    batch_size = nb_patches

    if prob != None :
        batch_size *= np.prod(1/prob) * 1.2

    # Explicit cast for future numpy requirement
    nb_patches = int(nb_patches)

    # Reserve memory for output arrays
    patches_label = np.empty((nb_patches, 1, label_patch_sz[0], label_patch_sz[1]), dtype = labels.dtype)
    patches_image = np.empty((nb_patches, nb_channels, image_patch_sz[0], image_patch_sz[1]), dtype = image.dtype)

    if image_patch_sz[0] < label_patch_sz[0] or image_patch_sz[1] < label_patch_sz[1]:
        raise NotImplementedError, "Image_patch_sz (={}) must be greater than label_patch_sz (={}) on both dimensions".format(image_patch_sz, label_patch_sz)

    (x,y,h,w) = crop_dimensions_from_patch_size(image_patch_sz, image.shape)

    current_ind = 0
    tm = [0.0]*4
    while current_ind < nb_patches:

        # Generate random indices
        tm_ind = 0
        t = time()
        this_batch_size = int(batch_size * float(nb_patches-current_ind) / float(nb_patches))
        col = np.random.randint(0, w, this_batch_size) + x
        lin = np.random.randint(0, h, this_batch_size) + y
        if label_patch_sz == (1,1):
            selected_labels = labels[lin, col][:, None, None, None]
        else:
            selected_labels = patch_at_location(labels, label_patch_sz, (lin, col))
        tm[tm_ind] += time()-t

        # Select patches based on class balance probability
        t=time()
        tm_ind += 1
        if prob != None:
            keep = np.ones(selected_labels.shape[0], dtype=np.bool)
            for cls_id, cls_prob in enumerate(prob):
                sel = (selected_labels == cls_id).any(axis=(1,2,3))
                keep[sel] = np.random.binomial(1, cls_prob, (keep[sel]).size).astype(np.bool)
            col = col[keep]
            lin = lin[keep]
            selected_labels = selected_labels[keep]
        nb_selected = selected_labels.shape[0]
        tm[tm_ind] += time()-t

        # Limit the number of selected patches (if more than nb_patches were selected)
        t=time()
        tm_ind += 1
        if nb_selected + current_ind > nb_patches:
            nb_selected = nb_patches - current_ind
            col = col[:nb_selected]
            lin = lin[:nb_selected]
            selected_labels = selected_labels[:nb_selected]
        tm[tm_ind] += time()-t

        # Extract the actual patches
        t=time()
        tm_ind += 1
        patches_label[current_ind:current_ind+nb_selected] = selected_labels
        patch_at_location(image, image_patch_sz, (lin,col), patches_image, current_ind)
        current_ind += nb_selected
        tm[tm_ind] += time()-t

    #print 'random_patches_with_replacement times = {}'.format(tm)

    assert current_ind==nb_patches

    return patches_image, patches_label


def patch_at_location(image, patch_sz, location, out = None, start_ind = 0):
    """
    Extract patches at specific locations.  For speed, one can pass a reference to an array and this function will copy the data
    straight into it.

    Parameters
    ----------
    image :         numpy.ndarray
                    image from which to extract patches

    patch_sz:       tuple of two integers
                    The size of patches to be extracted

    location:       tuple of two single dimension array like structures
                    indices of locations where to extract the patches.
                    location[0] corresponds to the patch line coordinate.
                    location[1] corresponds to the patch column coordinate.

    out:            [optional] numpy.ndarray
                    Output array of shape = (nb_patches, nb_channels, nb_lines, nb_columns)

    start_ind:      integer
                    index for which to start appending new patches in out

    Returns
    -------
    patches:        numpy.ndarray
                    Extracted patches of shape = (nb_patches, nb_channels, nb_lines, nb_columns).  If out is not None, then patches
                    is a reference to the out array
    """

    assert location[0].size == location[1].size

    nb_patches = len(location[0])
    nb_channels = 1 if image.ndim==2 else image.shape[2]

    if out is None:
        patches = np.empty((nb_patches, nb_channels, patch_sz[0], patch_sz[1]), dtype=image.dtype)
    else:
        patches = out

    (y,x,h,w) = crop_dimensions_from_patch_size(patch_sz, image.shape)

    llin = location[0]-y
    lcol = location[1]-x
    for i,(lin,col) in enumerate(zip(llin, lcol)):
        patches[i+start_ind] = image[lin:lin+patch_sz[0], col:col+patch_sz[1]][None, :, :]

    return patches


def ordered_patches(image, label=None, patch_sz = (11, 11), stride=1, type = 'regular'):
    """
    Extract ordered patched from a list ids of image.  Each patch is spaced by "stride" pixels.  Extraction goes
    from left to right and from top to bottom.

    Parameters
    ----------
    image:          numpy.ndarray of shape [MxNxC]
                    input image.  C corresponds the number of channels

    label:          numpy.ndarray of shape [MxN]
                    input label image.  It shape must be the same as image for the first two dimensions

    size:           tuple of two integers
                    The size of square patches to be extracted

    stride:         integer
                    The spacing between each patches in the extraction.  Currently, only a value of 1 is supported

    type:           string
                    "regular" or "conv"


    Returns
    -------
    patches:        numpy.ndarray
                    If type == conv, patches.shape = (nb_patches, nb_lines, nb_columns, nb_channels).
                    If type == regular, patches.shape = (nb_patches, nb_channels*nb_columns*nb_lines).
    patches_label : numpy.ndarray
                    Label associated to each patch patches_label.shape = (nb_patches,)
    """

    (y,x,h,w) = crop_dimensions_from_patch_size(patch_sz, image.shape, stride)

    nb_channels = 1 if image.ndim==2 else image.shape[2]

    # extract patches
    if type == "regular"   :
        patches_image = extract_patches_2d(image, patch_sz).reshape((-1, patch_sz[0]*patch_sz[1]*nb_channels))
    elif type == "conv":
        if image.ndim == 2:
            patches_image = extract_patches_2d(image, patch_sz)[:,None,:,:]
        else:
            patches_image = extract_patches_2d(image, patch_sz).transpose(0, 3, 1, 2)

    # extract labels
    patches_label = None
    if label != None:
        patches_label = label[y:y+h, x:x+w].flatten()

    return patches_image, patches_label, (y,x,h,w)


def crop_dimensions_from_patch_size(patch_sz, image_sz, stride=1):
    """
    In order to have consistency between modules, this function acts as reference for resulting size of an image for which
    patches were extracted

    Parameters
    ----------
    patch_sz:       tuple
                    patch size (height, width)

    image_sz:       tuple
                    image size (height, width)

    stride:         integer
                    stride width

    Returns
    -------
    (y,x,h,w)       tuple
                    cropped coordinate to that cropped_image = image[y:y+h, x:x+w]

    """
    y = int(floor(patch_sz[0]/2.))
    x = int(floor(patch_sz[1]/2.))
    h = int(floor((image_sz[0]-patch_sz[0])/ stride)+1)
    w = int(floor((image_sz[1]-patch_sz[1])/ stride)+1)

    return (y,x,h,w)


def vector_to_image(self, vec, dims):
    """
    Transform a vector into an image.   Each row of vec corresponds to a pixel.  Each column of vec corresponds to a channel
    of the image.  The size of the resulting image correspond to the cropped area of the

    Parameters
    ----------
    vec:            np.ndarray where vec.ndims == 1
                    input image in vector format

    dims:           tuple where len(dims)==2
                    image size (height, width)


    Returns
    -------
    image           np.ndarray
                    2D image where image.dtype == vec.dtype and where image.shape = dims
    """

    assert len(dims) == 2
    height = dims[0]
    width = dims[1]
    image = np.reshape(vec, (height, width))
    return image

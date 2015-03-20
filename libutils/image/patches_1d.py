# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:34:30 2013

@author: root
"""
import numpy                                as np
from math                                   import floor
from time                                   import time
from sklearn.feature_extraction.image       import extract_patches_2d

def random_patches(image, patch_sz, nb_patches):
    nb_channels = 1 if image.ndim==2 else image.shape[2]
    patches = extract_patches_2d(image, patch_sz, nb_patches, int(time()*10**6))
    return patches.reshape((-1, patch_sz[0]*patch_sz[1]*nb_channels))


def random_patches_without_replacement(image, labels, patch_sz, nb_patches, balance=None, mask=None):
    """
    Extract patches at random locations along with its corresponding labels.  This function ensures that the same patch will 
    never be selected twice.  It is recommended to use this function for small number of pixels (or with very sparse masks) as
    it becomes considerably slow as the number of patches increases.  
    
    IMPORTANT : Since no two same patche can be selected, this function does not guarantee that nb_patches will be returned.
    
    An optional balance parameter can be passed so that label i be chosen with a probability of balance[i].  This parameter was 
    added in order to tackle the inter-class imbalance problem.  See the following paper for details:
    
    An optional mask can be used, so that only pixels for which mask is nonzero are considered.  Mind that this function becomes
    considerabely slow as the number of zero in mask increases.
    
    Haibo He and Edwardo A. Garcia. 2009. Learning from Imbalanced Data. IEEE Trans. on Knowl. and Data Eng. 21, 9 
    (September 2009), 1263-1284. DOI=10.1109/TKDE.2008.239 http://dx.doi.org/10.1109/TKDE.2008.239 
    
    Parameters
    ----------
    image :         numpy.ndarray
                    image from which to extract patches
                    
    patch_sz:       tuple of two integers
                    The size of patches to be extracted

    location:       tuple of two single dimension array like structures
                    indices of locations where to extract the patches.
                    location[0] corresponds to the patch line coordinate
                    location[1] corresponds to the patch column coordinate
    """
    
    # Extract indices 
    lin,col = mask.nonzero()
    permutations = np.random.permutation(lin.size)
    lin = lin[permutations]
    col = col[permutations]
    
    patches_label = labels[lin, col]
    
    # Balance dataset
    if balance != None:
        keep = np.ones(patches_label.size, dtype=np.bool)
        for cls_id, cls_prob in enumerate(balance):
            keep[patches_label == cls_id] = np.random.binomial(1, cls_prob, (keep[patches_label == cls_id]).size).astype(np.bool)
        lin = lin[keep]
        col = col[keep]
        patches_label = patches_label[keep]
    
    # Keep only a sufficient number of patches
    if patches_label.size > nb_patches:
        lin = lin[:nb_patches]
        col = col[:nb_patches]
        patches_label = patches_label[:nb_patches]
        
    patches_image = patch_at_location(image, patch_sz, (lin,col))
    return patches_image, patches_label

 
def random_patches_with_replacement(image, labels, patch_sz, nb_patches, prob=None):
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
                    image from which to extract patches
                    
    patch_sz:       tuple of two integers
                    The size of patches to be extracted

    nb_patches:     integer
                    number of patches to be extracted

    prob:           [optional] numpy.ndarray 
                    Probability of selection for each class, such that the probability to select label[lin,col]==k is balance[k]
                    Therefore, if there is 4 possible labels then balance.shape == (number_of_labels,)
                    
    mask:           [optional] numpy.ndarray, dtype = numpy.bool
                    Can only select pixels (lin, col) for which mask[lin, col] is True
    """

    # Initializations

    nb_channels = 1 if image.ndim==2 else image.shape[2]
    batch_size = nb_patches

    if prob != None : 
        batch_size *= np.prod(1/prob) * 1.2
        
    # Explicit cast for future numpy requirement
    nb_patches = int(nb_patches)
          
    # Reserve memory for output arrays
    patches_label = np.empty(int(nb_patches), dtype = labels.dtype)
    patches_image = np.empty((nb_patches, patch_sz[0]*patch_sz[1]*nb_channels), dtype = image.dtype)
    
    (x,y,h,w) = crop_dimensions_from_patch_size(patch_sz, image.shape)

    current_ind = 0
    tm = [0.0]*4
    while current_ind < nb_patches:
        
        # Generate random indices
        tm_ind = 0
        t = time()
        this_batch_size = int(batch_size * float(nb_patches-current_ind) / float(nb_patches))
        col = np.random.randint(0, w, this_batch_size) + x
        lin = np.random.randint(0, h, this_batch_size) + y
        selected_labels = labels[lin, col]
        tm[tm_ind] += time()-t

        # Select patches based on class balance probability
        t=time()
        tm_ind += 1
        if prob != None:
            keep = np.ones(selected_labels.size, dtype=np.bool)
            for cls_id, cls_prob in enumerate(prob):
                tmp_sel = selected_labels == cls_id
                keep[tmp_sel] = np.random.binomial(1, cls_prob, (keep[tmp_sel]).size).astype(np.bool)
            col = col[keep]
            lin = lin[keep]
            selected_labels = selected_labels[keep]
        nb_selected = selected_labels.size 
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
        patch_at_location(image, patch_sz, (lin,col), patches_image, current_ind)
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
                    Output array of shape = (nb_patches, nb_dimensions)
    
    start_ind:      integer
                    index for which to start appending new patches in out
                    
    Returns
    -------
    patches:        numpy.ndarray
                    Extracted patches of shape = (nb_patches, nb_dimensions).  If out is not None, then patches is a reference to
                    the out array
    """
    
    assert location[0].size == location[1].size

    nb_patches = len(location[0])
    nb_channels = 1 if image.ndim==2 else image.shape[2]

    if out is None:
        patches = np.empty((nb_patches, patch_sz[0]*patch_sz[1]*nb_channels), dtype=image.dtype)
    else:
        patches = out
        
    (y,x,h,w) = crop_dimensions_from_patch_size(patch_sz, image.shape)

    llin = location[0]-y
    lcol = location[1]-x
    for i,(lin,col) in enumerate(zip(llin, lcol)):
        patches[i+start_ind] = image[lin:lin+patch_sz[0], col:col+patch_sz[1]].flatten()
    
    return patches


def ordered_patches(image, label=None, patch_sz = 10, stride=1):
    """
    Extract ordered patched from a list ids of image.  Each patch is spaces by "stride" pixels.  Extraction goes
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
    """
    
    (y,x,h,w) = crop_dimensions_from_patch_size(patch_sz, image.shape, stride)

    nb_channels = 1 if image.ndim==2 else image.shape[2]

    # extract patches    
    patches_image = extract_patches_2d(image, patch_sz).reshape((-1, patch_sz[0]*patch_sz[1]*nb_channels))
    
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
    
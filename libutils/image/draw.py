# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 13:55:56 2014

@author: francis
"""

import numpy                        as np
from scipy                          import ndimage
from skimage.draw                   import line_aa


def rectangle_center(img, rect, rot, color, thick = 1, filled = False):
    """
    Draws a rectangle given it's center position, width and height and rotation
    angle.
    
    Notes
    -----
    If filled is true, than the thickness parameter is not used.    
    
    If filled is False and thickness is greater than one, then the drawn 
    rectangle is grown outward.  
    
    
    if the specified     
    
    Parameters
    ----------
    img:        numpy.ndarray
                input image with shape [nb_lin, nb_col, nb_channels]
                
    rect:       tuple of integers with format (y, x, h, w)
                center position (y,x) and size (h,w) of the rectangle 
                
    rot:        decimal type in radians
                Rotation angle of the rectangle.  The rotation axis is located
                at the centroid of the rectangle.
                
    color:      array like type
                Rectangle color.  Size of array must be equal to nb_channels of 
                input img
                
    thick:      integer type [default = 1]
                Rectangle thickness.  
                
    Returns
    -------
    out         numpy.ndarray
                img with drawn rectangle
    """
    
    # Rectangle center position
    r_y = rect[0]    
    r_x = rect[1]
    
    # Rectangle width and height
    r_h = rect[2]
    r_w = rect[3]
    
    # Image center coordinates
    i_y = img.shape[0] // 2
    i_x = img.shape[1] // 2

    
    if filled : 
        # Centered inner filled rectangle
        rect = np.zeros(img.shape[0:2], dtype=np.float)
        rect[i_y-r_h//2:i_y-r_h//2+r_h, i_x-r_w//2:i_x-r_w//2+r_w] = 1
    
    else:
        # Centered outer filled rectangle
        rect = np.zeros(img.shape[0:2], dtype=np.float)
        rect[i_y-r_h//2-thick:i_y-r_h//2+r_h+thick, i_x-r_w//2-thick:i_x-r_w//2+r_w+thick] = True
        rect[i_y-r_h//2+1    :i_y-r_h//2+r_h-1,     i_x-r_w//2+1    :i_x-r_w//2+r_w-1] = False
        
    # Rotate rectangle if necessary
    if rot % np.pi != 0:
        rect[:] = ndimage.rotate(rect, rot * 180. / np.pi, reshape = False, order=1)
    
    # Shift rectangle to its destination
    rect[:] = ndimage.shift(rect, (r_y-i_y, r_x-i_x), order=1)
    
    # Apply rectangle coloration to image
    out = img.copy()
    sel = rect > 0.01
    color = np.asarray(color)[None, :]
    out[sel] = (1-rect[sel])[:,None]*out[sel] + rect[sel][:,None]*color
    
    return out
    


def rectangle_corner(img, rect, color, thick = 1, filled = False):
    """
    Draws a rectangle using top left corner coordinates.
    
    Notes
    -----
    If filled is true, than the thickness parameter is not used.    
    
    If filled is False and thickness is greater than one, then the drawn 
    rectangle is grown outward.  
    
    
    if the specified     
    
    Parameters
    ----------
    img:        numpy.ndarray
                input image with shape [nb_lin, nb_col, nb_channels]
                
    rect:       tuple of integers with format (y, x, h, w)
                center position (y,x) and size (h,w) of the rectangle 
                
    color:      array like type
                Rectangle color.  Size of array must be equal to nb_channels of 
                input img
                
    thick:      integer type [default = 1]
                Rectangle thickness.  
                
    Returns
    -------
    out         numpy.ndarray
                img with drawn rectangle
    """

    # Rectangle center position
    r_y = rect[0]    
    r_x = rect[1]
    
    # Rectangle width and height
    r_h = rect[2]
    r_w = rect[3]
    
    if filled : 
        rect = np.zeros(img.shape[0:2], dtype=np.float)
        rect[r_y:r_y+r_h, r_x:r_x+r_w] = True
    
    else:
        rect = np.zeros(img.shape[0:2], dtype=np.float)
        rect[r_y-thick:r_y+r_h+thick, r_x-thick:r_x+r_w+thick] = True
        rect[r_y+1:r_y+r_h-1, r_x+1:r_x+r_w-1] = False
        
    # Apply rectangle coloration to image
    out = img.copy()
    sel = rect > 0.01
    color = np.asarray(color)[None, :]
    out[sel] = (1-rect[sel])[:,None]*out[sel] + rect[sel][:,None]*color
    
    return out
    
    


def line(img, coord, color):
    """
    Draws a line on the input image
    
    Parameters
    ----------
    img:        numpy.ndarray
                input image with shape [nb_lin, nb_col, nb_channels]
                
    coord:      tuple of integers with format (y1, x1, y2, x2)
                position (before rotation) and size of the rectangle 
                
    color:      array like type
                Rectangle color.  Size of array must be equal to nb_channels of 
                input img

    Returns
    -------
    out         numpy.ndarray
                img with drawn rectangle
    """

    # Line coordinates
    y1, x1, y2, x2 = coord
    
    # Generate line coordinates
    rr, cc, val = line_aa(y1, x1, y2, x2)
    
    
    # Draw line color
    out = img.copy()
    val = val[:, None]
    color = np.asarray(color)[None, :]
    print "(out[rr, cc] * (1 - val)).shape = ", (out[rr, cc] * (1 - val)).shape
    print "(out[rr, cc] * val * color).shape = ", (out[rr, cc] * val * color).shape
    out[rr, cc] = out[rr, cc] * (1 - val) +  out[rr, cc] * val * color
    
    return out
    

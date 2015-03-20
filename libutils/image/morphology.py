# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 11:22:09 2014

@author: francis
"""

import cv2
import numpy as np
from blobs import findblobs

__structuring_elm = {'circular': cv2.MORPH_ELLIPSE, 'square':cv2.MORPH_RECT}

def __preprocess(img):
    return img.astype(np.uint8) if img.dtype != np.uint8 else img

def imerode(img, kernel_size=3, kernel_type = 'circular'):  
    return cv2.erode(__preprocess(img), cv2.getStructuringElement(__structuring_elm[kernel_type], (kernel_size,kernel_size)))

def imdilate(img, kernel_size, kernel_type = 'circular'):  
    return cv2.dilate(__preprocess(img), cv2.getStructuringElement(__structuring_elm[kernel_type], (kernel_size,kernel_size)))

def imopen(img, kernel_size, kernel_type = 'circular', nb_iterations=1):
    return cv2.morphologyEx(__preprocess(img), cv2.MORPH_OPEN, cv2.getStructuringElement(__structuring_elm[kernel_type], (kernel_size,kernel_size)), iterations=nb_iterations)

def imclose(img, kernel_size, kernel_type = 'circular', nb_iterations=1):
    return cv2.morphologyEx(__preprocess(img), cv2.MORPH_CLOSE, cv2.getStructuringElement(__structuring_elm[kernel_type], (kernel_size,kernel_size)), iterations=nb_iterations)

def tophat(img, kernel_size, kernel_type = 'circular', nb_iterations=1):
    return cv2.morphologyEx(__preprocess(img), cv2.MORPH_TOPHAT, cv2.getStructuringElement(__structuring_elm[kernel_type], (kernel_size,kernel_size)), iterations=nb_iterations)

def blackhat(img, kernel_size, kernel_type = 'circular', nb_iterations=1):
    return cv2.morphologyEx(__preprocess(img), cv2.MORPH_BLACKHAT, cv2.getStructuringElement(__structuring_elm[kernel_type], (kernel_size,kernel_size)), iterations=nb_iterations)
    
def imfill(img):
    img = img.astype(np.uint8)
    blobs = findblobs(img)
    for b in blobs:
        cv2.drawContours(img, [b.contour], 0, 255, -1)
    return img

def imclean(img, object_to_remove_max_size):
    img = img.astype(np.uint8)
    blobs = findblobs(img)
    for i, b in enumerate(blobs):
        if b.area <= object_to_remove_max_size:
            cv2.drawContours(img, [b.contour], 0, 0, -1)
    return img

if __name__ == '__main__':
    
    import sys
    sys.path.append("../")

    from utils.matplotlib import imshow, subplots
    from os.path import join
    
    path = '../../datasets/cells/hematology/staging/images'
        
    img = cv2.imread(join(path, 'Giemsa_Halogene_2L-G514308471_wbc_1.tif'))[:,:,::-1]/255.
    bw = img[:,:,0] < 0.9

    sp = subplots(3, 4)
    imshow(sp[0,0], img, title='original')
    imshow(sp[0,1], bw, title='bin')
    imshow(sp[0,2], imfill(bw) , title='filled')
    imshow(sp[0,3], imclean(bw, 30), title='filled')
    imshow(sp[1,0], imerode(bw, 5), title='erode')
    imshow(sp[1,1], imdilate(bw, 5), title='dilate')
    imshow(sp[1,2], imopen(bw, 5), title='open')
    imshow(sp[1,3], imclose(bw, 5), title='close')
    imshow(sp[2,0], img[:,:,0], title='channel r')
    imshow(sp[2,1], tophat(img[:,:,0]*255, 21), title='tophat')
    imshow(sp[2,2], blackhat(img[:,:,0]*255, 21), title='blackhat')    
    sp.save('test_morphology.png')
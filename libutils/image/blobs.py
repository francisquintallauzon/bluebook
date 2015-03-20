# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 16:14:16 2014

@author: francis
"""

import cv2
import numpy as np



class blob(object):
    def __init__(self, idx, contours, hierarchy):
        
        self.idx = idx
        self.contours = contours
        self.hierarchy = hierarchy

        # Initialize attributes
        self.__area = None
        self.__moments = None
        self.__centroid_x = None
        self.__centroid_y = None
        self.__bb = None # Bounding box
        self.__image = None # Blob image
        self.__nested = None
        
    @property
    def contour(self):
        return self.contours[self.idx]
    
    @property
    def nested(self):
        if self.__nested is None:
            def nested(index):
                nested_contours = []
                while index >= 0:
                    nested_contours += nested(self.hierarchy[0,index,2])
                    nested_contours += [self.contours[index]]
                    if self.hierarchy[0,index,3] < 0:  # Top contour, we're only interested to childs
                        return nested_contours
                    index = self.hierarchy[0,index,0]
                return nested_contours
            self.__nested = nested(self.idx)
        return self.__nested

    @property
    def area(self):
        if self.__area is None:
            self.__area = 0
            for m in self.moments:
                self.__area += m['m00']
        return self.__area
        
    @property
    def moments(self):
        if self.__moments is None:
            self.__moments = []
            for cnt in self.nested:
                moment = cv2.moments(cnt)
                area = -cv2.contourArea(cnt, True)
                moment['m00'] = area + (1 if area==0 else 0)
                moment['m10'] = ((1 if area>0 else -1) * moment['m10']) if area else cnt[0,0,0]
                moment['m01'] = ((1 if area>0 else -1) * moment['m01']) if area else cnt[0,0,1]
                self.__moments += [moment]
        return self.__moments
                 
    @property
    def centroid_x(self):
        if self.__centroid_x is None:
            moment = 0
            #area = 0
            for m in self.moments:
                moment += m['m10']
                #area += m['m00']
            self.__centroid_x = moment / self.area
        return self.__centroid_x

    @property
    def centroid_y(self):
        if self.__centroid_y is None:
            moment = 0
            #area = 0
            for m in self.moments:
                moment += m['m01']
                #area += m['m00']
            self.__centroid_y = moment / self.area
        return self.__centroid_y

    @property
    def bb(self):
        if self.__bb is None:
            cnt = self.contours[self.idx][:,0,:]
            x = int(cnt[:,0].min())
            y = int(cnt[:,1].min())
            w = int(cnt[:,0].max() - x + 1)
            h = int(cnt[:,1].max() - y + 1)
            self.__bb = [x, y, w, h]
        return self.__bb
    
    @property    
    def image(self):
        if self.__image is None:
            self.__image = np.zeros((self.bb[3], self.bb[2]), np.uint8)
            offset_cnt = []
            for cnt in self.contours:
                offset_cnt += [cnt - np.asarray((self.bb[0], self.bb[1]))[None, None, :]]
            cv2.drawContours(self.__image, offset_cnt, self.idx, 255, -1, hierarchy=self.hierarchy)
            self.__image = self.__image != 0
        return self.__image


class findblobs(object):

    def __init__(self, img):
        contours, hierarchy = cv2.findContours(img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.__blobs = [] if hierarchy is None else [blob(i, contours, hierarchy) for i, h in enumerate(hierarchy[0]) if h[3]<0]
        
    def __len__(self):
        return len(self.__blobs)
        
    def __getitem__(self, key):
        return self.__blobs[key]
        
    def __getattr__(self, key):
        return [getattr(m, key) for m in self.__blobs]
        
    def __iter__(self):
        for m in self.__blobs:
            yield m


if __name__ == '__main__':
    
    import sys
    sys.path.append("../")

    from utils.matplotlib import imshow, subplots
    from os.path import join
    import numpy as np
    
    path = '../../datasets/cells/hematology/staging/images'
        
    img = cv2.imread(join(path, 'Giemsa_Halogene_2L-G514308471_wbc_1.tif'))[:,:,::-1]/255.
    bw = img[:,:,0] < 0.9

    blobs = findblobs(bw)
        
    sp = subplots(1, 2, 3, 6)
    imshow(sp[0], img, title='original')
    imshow(sp[1], bw, title='original')
    for b in blobs:
        print b.area, (b.centroid_x, b.centroid_y)
        sp[1].annotate('{}'.format(b.area), xy=(b.centroid_x, b.centroid_y), color='r', fontsize=2)
    print 'Almost done... saving output image!'
    sp.save('test_blobs.png')
    print 'Done!'
    


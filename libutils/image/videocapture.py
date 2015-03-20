# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 10:54:08 2014

@author: francis
"""

import cv2

class videocapture(object):
    """
    Minimal wrapper class for opencv VideoCapture
    """

    def __init__(self, filename):
        self.filename = filename
        self.__v = cv2.VideoCapture(filename)

    @property
    def nb_frames(self):
        """ Returns the video's number of frames """
        return int(self.__v.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    @property
    def frame_position(self):
        """ Returns the current position of the next frame to decode """
        return int(self.__v.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))

    @frame_position.setter
    def frame_position(self, pos):
        """
        Sets the position for the next frame to decode
        """
        if pos < 0 or pos > self.nb_frames:
            raise ValueError, 'Can set to frame position to {}.  Total number of frames in video = {}'.format(pos, self.nb_frames)

        self.__v.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos)

        if self.frame_position != pos :
            raise IOError, 'Could not set to frame position to {} in {}.  \n Total number of frames in video = {}.  Current frame position {}'.format(pos, self.filename, self.nb_frames, self.frame_position)


    @property
    def width(self):
        """ Returns the video frame width """
        return int(self.__v.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))

    @property
    def height(self):
        """ Returns the video frame height """
        return int(self.__v.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

    def read(self):
        """
        Reads the next video frame
        """
        return self.__v.read()




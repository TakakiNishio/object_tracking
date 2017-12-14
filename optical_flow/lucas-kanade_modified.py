#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='point tracking')
parser.add_argument('--video_file', '-v', type=str, default=False,help='path to video')
args = parser.parse_args()

# ESC_KEY = 0x1b # Esc key
# S_KEY = 0x73 # s key
# R_KEY = 0x72 # r key

ESC_KEY = 27 # Esc key
S_KEY = ord("s") # s key
R_KEY = ord("r") # r key

MAX_FEATURE_NUM = 500 # maximum amount of feature points
CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03) # finish condition of iteration

INTERVAL = 30 # interval (1000 / frame rate)
VIDEO_DATA = args.video_file

class Motion:
    # constructor
    def __init__(self):
        # display window
        cv2.namedWindow("motion")
        # video
        self.video = cv2.VideoCapture(VIDEO_DATA)
        # interval
        self.interval = INTERVAL
        # current frame (RGB)
        self.frame = None
        # current frame (GRAY)
        self.gray_next = None
        # previous frame (GRAY)
        self.gray_prev = None
        # feature points
        self.features = None
        # status of feature points
        self.status = None


    # main loop
    def run(self):

        # process of initial frame
        end_flag, self.frame = self.video.read()
        self.gray_prev = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        while end_flag:

            # load point
            self.load_point()

            # convert the frame into gray scale
            self.gray_next = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            # calculate OpticalFlow when the feature points is registered
            if self.features is not None:
                # calculation of OpticalFlow
                features_prev = self.features
                self.features, self.status, err = cv2.calcOpticalFlowPyrLK( \
                                                    self.gray_prev, \
                                                    self.gray_next, \
                                                    features_prev, \
                                                    None, \
                                                    winSize = (10, 10), \
                                                    maxLevel = 3, \
                                                    criteria = CRITERIA, \
                                                    flags = 0)

                # leave valid feature points
                self.refreshFeatures()

                # draw valid feature points to current frame
                if self.features is not None:
                    for feature in self.features:
                        cv2.circle(self.frame, (feature[0][0], feature[0][1]), 4, (15, 241, 255), -1, 8, 0)

            # display
            cv2.imshow("motion", self.frame)

            # preparation of next loop process
            self.gray_prev = self.gray_next
            end_flag, self.frame = self.video.read()
            if end_flag:
                self.gray_next = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            # interval
            key = cv2.waitKey(self.interval) & 0xFF
            # finish program when ESC key
            if key == ESC_KEY:
                break
            # pause program when S key
            elif key == S_KEY:
                self.interval = 0
            # restart program when S key
            elif key == R_KEY:
                self.interval = INTERVAL


        # finish process
        cv2.destroyAllWindows()
        self.video.release()


    # specify the feature point by mouse click
    #   delete the feature point if there is another feature point near clicked point
    #   add the feature point if there is no another feature point near clicked point
    def load_point(self):
        x = 190
        y = 190

        # add initial feature point
        if self.features is None:
            self.addFeature(x, y)
            return

        # sarching radius (pixel)
        radius = 5
        # sarch near existing feature points
        index = self.getFeatureIndex(x, y, radius)

        # delete existing feature point if there is another feature point near clicked point
        if index >= 0:
            self.features = np.delete(self.features, index, 0)
            self.status = np.delete(self.status, index, 0)

        # add the feature point if there is no another feature point near clicked point
        else:
            self.addFeature(x, y)

        return


    # obtain an index of existing feature point in specified radius
    # return index = -1 if there is no feature point in specified radius
    def getFeatureIndex(self, x, y, radius):
        index = -1

        # no registeration of feature points
        if self.features is None:
            return index

        max_r2 = radius ** 2
        index = 0
        for point in self.features:
            dx = x - point[0][0]
            dy = y - point[0][1]
            r2 = dx ** 2 + dy ** 2
            if r2 <= max_r2:
                # this feature point is in the specified radius
                return index
            else:
                # this feature point is out of the specified radius
                index += 1

        # all feature points are out of the specified radius
        return -1


    # add new feature point
    def addFeature(self, x, y):

        # if no feature points are registered
        if self.features is None:
            # create ndarray and register the coordinate of feature point
            self.features = np.array([[[x, y]]], np.float32)
            self.status = np.array([1])
            # make the feature point more presise
            cv2.cornerSubPix(self.gray_next, self.features, (10, 10), (-1, -1), CRITERIA)

        # over maximum amount of feature point registerations
        elif len(self.features) >= MAX_FEATURE_NUM:
            print("max feature num over: " + str(MAX_FEATURE_NUM))

        # add feature point registeration
        else:
            # add feature point coordinate to the end of existing ndarray
            self.features = np.append(self.features, [[[x, y]]], axis = 0).astype(np.float32)
            self.status = np.append(self.status, 1)
            # make the feature point more presise
            cv2.cornerSubPix(self.gray_next, self.features, (10, 10), (-1, -1), CRITERIA)


    # leave only valid feature point
    def refreshFeatures(self):
        # if there is no registeration of feature points
        if self.features is None:
            return

        # check all status
        i = 0
        while i < len(self.features):

            # if the status can not be recognize as feature point
            if self.status[i] == 0:
                # delete the status from existing ndarray
                self.features = np.delete(self.features, i, 0)
                self.status = np.delete(self.status, i, 0)
                i -= 1

            i += 1


if __name__ == '__main__':

    #Motion().run()
    tracking_function = Motion()
    tracking_function.run()

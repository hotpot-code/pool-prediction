from collections import deque
import numpy as np
import cv2
import math
import imutils
import matplotlib.pyplot as plt

class BallDetection():

    def __init__(self, colorMin, colorMax, radiusMin, radiusMax):
        self.colorMin = colorMin
        self.colorMax = colorMax
        self.radiusMin = radiusMin
        self.radiusMax = radiusMax
        self.last_x = None
        self.last_y = None

    def detectBall(self, frame):
        #blur image
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        #convert to hsv
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)


        mask = cv2.inRange(hsv, self.colorMin, self.colorMax)
        
        # clean mask
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=1)
        
        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        # only proceed if at least one contour was found
        if len(contours) > 0:
            
            # if multiple contours found and we get the last position
            if len(contours) > 1 and self.last_x is not None and self.last_y is not None:
                # find nearest contour compared to the last position
                min_distance = math.inf
                for contour in contours:
                    ((x, y), radius) = cv2.minEnclosingCircle(contour)
                    distance_to_last = math.sqrt(math.pow(self.last_x - x, 2) + math.pow(self.last_y - y, 2))
                    if distance_to_last < min_distance:
                        min_distance = distance_to_last
                        c = contour
            else:
                c = contours[0]
            
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            if radius > self.radiusMin and radius < self.radiusMax:
                self.last_x = x
                self.last_y = y
                return (x,y)
        
        self.last_x = None
        self.last_y = None
        return (None, None)
    


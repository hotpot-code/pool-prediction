# import the necessary packages
from collections import deque
import numpy as np
import cv2
import math
import imutils
import matplotlib.pyplot as plt

from BallDetection import BallDetection 

from filter import MyFilter

kalman = MyFilter(0.03333)

    
# lower and upper boundaries of the "white"
whiteLower = (10, 1, 1)
whiteUpper = (50, 120, 250)
whiteBallDetection = BallDetection(whiteLower, whiteUpper, 5, 11)

# load video
vs = cv2.VideoCapture("videos/pool.mp4")

frame_no = 0

# keep looping
while True:
    
    # Get frame from video
    frame = vs.read()
    frame = frame[1]
 
    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if frame is None:
        break
        
    # crop image
    frame = frame[70:620, 100:1150]
    # resize image
    frame = imutils.resize(frame, width=600)

    x,y = whiteBallDetection.detectBall(frame)
    
    if (x is not None and y is not None):
        cv2.circle(frame, (int(x), int(y)), int(20), (255, 255, 255), 2)

    filterd = kalman.dofilter(x, y)
    #cv2.circle(frame_copy, (int(filterd[0]), int(filterd[1])), int(15), (255, 255, 0), 2)
    
    # show the frame to our screen
    cv2.imshow("Frame", frame)
    cv2.waitKey(10)

    
cv2.waitKey(2000)
cv2.destroyWindow("Frame")
cv2.waitKey(2000)
vs.release()
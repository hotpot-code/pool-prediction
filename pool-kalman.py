# import the necessary packages
from collections import deque
import numpy as np
import copy 
import cv2
import math
import imutils
import matplotlib.pyplot as plt

from BallDetection import BallDetection 

from filter import MyFilter

kalman = MyFilter(0.03333, 20.0)
    
# lower and upper boundaries of the "white"
whiteLower = (10, 1, 1)
whiteUpper = (50, 120, 250)
whiteBallDetection = BallDetection(whiteLower, whiteUpper, 5, 11)

# load video
vs = cv2.VideoCapture("videos/pool.mp4")

frame_no = 0

last_points_filtered = deque([])
last_points = deque([])
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
        last_points.append([x,y])

    if len(last_points) > 1:
        last_point = None
        for point in last_points:
            if last_point is not None:
                cv2.line(frame, (int(last_point[0]),int(last_point[1])), (int(point[0]),int(point[1])), (0, 255, 255), 2)
            last_point = point

    last_point = None
    filterd = kalman.dofilter(x, y)
    last_points_filtered.append(filterd)
    if len(last_points_filtered) > 1:
        last_point = None
        for point in last_points_filtered:
            if last_point is not None:
                cv2.line(frame, (int(last_point[0]),int(last_point[1])), (int(point[0]),int(point[1])), (0, 255, 0), 2)
            last_point = point
            

    prediction = kalman.getPredictionAfterSec(0.33)
    cv2.line(frame, (int(filterd[0]),int(filterd[1])), (int(prediction[0]),int(prediction[1])), (0, 0, 255), 2)

        
    cv2.circle(frame, (int(filterd[0]), int(filterd[1])), 2, (255, 255, 0), 2)
    
    # show the frame to our screen
    cv2.imshow("Frame", frame)
    cv2.waitKey(20)
    frame_no += 1

    
cv2.waitKey(2000)
cv2.destroyWindow("Frame")
cv2.waitKey(2000)
vs.release()
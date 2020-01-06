# import the necessary packages
from collections import deque
import numpy as np
import copy 
import cv2
import math
import imutils
import matplotlib.pyplot as plt
import random

from BallDetection import BallDetection 

from filter.filter_constant_velocity import MyFilter

kalman = MyFilter(0.01666, 600.0, 2.1)
    
# lower and upper boundaries of the "white"
whiteLower = (20, 0, 150)
whiteUpper = (45, 165, 255)
whiteBallDetection = BallDetection(whiteLower, whiteUpper, 3, 11)

# load video
vs = cv2.VideoCapture("videos/pool_4.mp4")

frame_no = 0

last_points_filtered = deque([])
last_points = deque([])

abweichung_x = 0
abweichung_y = 0

#frame_number = 360 #pool_3 and others
frame_number = 30 #pool_4

vs.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1);

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
    #frame = frame[60:620, 100:1150]
    # resize image
    frame = imutils.resize(frame, width=600)

    x,y = whiteBallDetection.detectBall(frame)
    x_correct = x
    y_correct = y

    if (x is not None and y is not None):
        x = x + np.random.randn() * 2
        y = y + np.random.randn() * 2
    
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
            
    kalmanPrediction = copy.deepcopy(kalman)
    last_prediction = filterd
    for i in range(0, 50):
        new_prediction = kalmanPrediction.dofilter(None, None)
        cv2.line(frame, (int(last_prediction[0]),int(last_prediction[1])), (int(new_prediction[0]),int(new_prediction[1])), (0, 0, 255), 2)
        last_prediction = new_prediction

    #prediction = kalman.getPredictionAfterSec(0.33)
    #cv2.line(frame, (int(filterd[0]),int(filterd[1])), (int(prediction[0]),int(prediction[1])), (0, 0, 255), 2)

    # if x is not None:
    #     abweichung_x += abs(x_correct - x)
    #     abweichung_y += abs(y_correct - y)

    #     print("abweichung x: " + str(abweichung_x/(frame_no + 1)))
    #     print("abweichung y: " + str(abweichung_y/(frame_no + 1)))

        
    cv2.circle(frame, (int(filterd[0]), int(filterd[1])), 2, (255, 255, 0), 2)
    
    # show the frame to our screen
    cv2.imshow("Frame", frame)
    cv2.waitKey(20)
    frame_no += 1

    
cv2.waitKey(2000)
cv2.destroyWindow("Frame")
cv2.waitKey(2000)
vs.release()
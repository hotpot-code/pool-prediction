# import the necessary packages
from collections import deque
import numpy as np
import copy 
import cv2
import math
import matplotlib.pyplot as plt
import random

from BallDetection import BallDetection 
from VideoHandler import VideoHandler

from filter.filter import MyFilter

kalman = MyFilter(0.01666, 600.0, 2.1)

vh = VideoHandler("pool_5")
whiteBallDetection = BallDetection(*vh.giveParameters())

frame_no = 0

last_points_filtered = deque([])
last_points = deque([])

abweichung_x = 0
abweichung_y = 0

# keep looping
while True:
    
    frame = vh.giveFrames()
 
    ## if we are viewing a video and we did not grab a frame,
    ## then we have reached the end of the video
    if frame is None:
        break
        
    ## crop and resize
    frame = vh.cropFrame(frame)

    x,y = whiteBallDetection.detectBall(frame)
    x_correct = x
    y_correct = y

    # if (x is not None and y is not None):
    #     x = x + np.random.randn() * 2
    #     y = y + np.random.randn() * 2
    
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

    velocity = np.array([
        [kalman.x_post[1]],
        [kalman.x_post[3]]
    ])
    tempo = np.linalg.norm(velocity)
    # 170cm / 600px
    tempoPerMeter = tempo * 0.028
    tempoRounded = int(tempoPerMeter * 100) / 100

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,30)
    fontScale              = 0.5
    fontColor              = (255,255,255)
    lineType               = 2

    cv2.putText(frame,str(tempoRounded) + ' m/s', 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

    #print(tempo)
    last_points_filtered.append(filterd)
    if len(last_points_filtered) > 1:
        last_point = None
        for point in last_points_filtered:
            if last_point is not None:
                cv2.line(frame, (int(last_point[0]),int(last_point[1])), (int(point[0]),int(point[1])), (0, 255, 0), 2)
            last_point = point
            
    prePos, preVar = kalman.getPredictions(10)
    for i in range(0, len(prePos)):
        cv2.ellipse(frame, (prePos[i][0], prePos[i][1]), (int(1* np.sqrt(preVar[i][0])), int(1*np.sqrt(preVar[i][1]))), 0, 0, 360, (0, 200, 255), 2)

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
vh.vs.release()
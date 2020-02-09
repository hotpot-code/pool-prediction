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

from filter.smart_filter import Smart_CAM_Filter
from filter.filter_constant_velocity import CVM_Filter
from filter.smart_cvm_filter import Smart_CVM_Filter


def residual(points, ground_truth):
    residuals = list()
    for i in range(0, len(points)):
        point = points[i]
        gt = ground_truth[i]
        if point[0] is not None and gt[0] is not  None:
            residuals.append((point[0] - gt[0], point[1] - gt[1]))
    r = np.array(residuals)
    rx = r[:, 0]
    ry = r[:, 1]
    mse = (rx ** 2 + ry ** 2).mean()
    return mse

vh = VideoHandler("pool_3")

p1 = (vh.xLeft, vh.yTop) #top left
p2 = (vh.xRight, vh.yTop) #top right
p3 = (vh.xRight, vh.yBot) # bottom right
p4 = (vh.xLeft, vh.yBot) # bottom left

#kalman = MyFilter(0.01666, 600.0, 0.001)
#pool3
kalman = Smart_CVM_Filter(0.016, 600, 0.25)
#pool1
#kalman = Smart_CVM_Filter(0.033, 500, 0.1)
#kalman = Smart_CAM_Filter(0.016, 820, 0.25)

kalman.setBoundaries(vh.xLeft, vh.xRight, vh.yTop, vh.yBot)


whiteBallDetection = BallDetection(*vh.giveParameters())

frame_no = 0

last_points_filtered = list()
last_points = list()

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

    x,y,radius = whiteBallDetection.detectBall(frame)

    x_correct = x
    y_correct = y

    # if (x is not None and y is not None):
    #     x = x + np.random.randn() * 2
    #     y = y + np.random.randn() * 2
    
    if (x is not None and y is not None):
        cv2.circle(frame, (int(x), int(y)), int(20), (255, 255, 255), 2)

    last_points.append([x, y])

    if len(last_points) > 1:
        last_point = None
        for point in last_points:
            if last_point is not None and point[0] is not None and last_point[0] is not None:
                cv2.line(frame, (int(last_point[0]),int(last_point[1])), (int(point[0]),int(point[1])), (0, 255, 255), 2)
            last_point = point

    last_point = None
    filterd = kalman.dofilter(x, y, radius)

    velocity = np.array([
        [kalman.x_post[1]],
        [kalman.x_post[3]]
    ])

    last_points_filtered.append(filterd)
    if len(last_points_filtered) > 1:
        last_point = None
        for point in last_points_filtered:
            if last_point is not None:
                cv2.line(frame, (int(last_point[0]),int(last_point[1])), (int(point[0]),int(point[1])), (0, 255, 0), 2)
            last_point = point
            
    prePos, preVar = kalman.getPredictions(60)
    for i in range(0, len(prePos), 2):
        cv2.ellipse(frame, (prePos[i][0], prePos[i][1]), (int(1 * np.sqrt(preVar[i][0])), int(1 * np.sqrt(preVar[i][1]))), 0, 0, 360, (0, 200, 255), 2)

    #prediction = kalman.getPredictionAfterSec(0.33)
    #cv2.line(frame, (int(filterd[0]),int(filterd[1])), (int(prediction[0]),int(prediction[1])), (0, 0, 255), 2)

    # if x is not None:
    #     abweichung_x += abs(x_correct - x)
    #     abweichung_y += abs(y_correct - y)

    #     print("abweichung x: " + str(abweichung_x/(frame_no + 1)))
    #     print("abweichung y: " + str(abweichung_y/(frame_no + 1)))

        
    cv2.circle(frame, (int(filterd[0]), int(filterd[1])), 2, (255, 255, 0), 2)

    cv2.line(frame, p1, p2, (255, 255, 255), 2) #top bank
    cv2.line(frame, p2, p3, (255, 255, 255), 2) #right bank
    cv2.line(frame, p3, p4, (255, 255, 255), 2) #bottom bank
    cv2.line(frame, p4, p1, (255, 255, 255), 2) #left bank 
    
    # show the frame to our screen
    cv2.imshow("Frame", frame)
    cv2.waitKey(14)
    frame_no += 1

    
cv2.waitKey(1000)
cv2.destroyWindow("Frame")
cv2.waitKey(1000)
vh.vs.release()

print(10 * math.log10(residual(last_points_filtered, last_points)))
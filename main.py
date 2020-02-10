import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import random

from BallDetection import BallDetection 
from VideoHandler import VideoHandler

from filter.smart_filter import Smart_CAM_Filter
from filter.filter_constant_velocity import CVM_Filter
from filter.smart_cvm_filter import Smart_CVM_Filter


# calculate mse of two point arrays
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


# get mse of prediction
def get_mse_of_prediction(predictions, points, pre_no = 10, offset=30):
    pre_pos = np.array(predictions)[offset:-pre_no, 0, pre_no]
    points = np.array(points)[offset+pre_no:]
    return 10 * np.log10(residual(pre_pos, points))

# load pool video
vh = VideoHandler("pool_3")

# get cushion coordinates
p1 = (vh.xLeft, vh.yTop) #top left
p2 = (vh.xRight, vh.yTop) #top right
p3 = (vh.xRight, vh.yBot) # bottom right
p4 = (vh.xLeft, vh.yBot) # bottom left

# instantiate kalmanfilter

#kalman = MyFilter(0.01666, 600.0, 0.001)
#pool3
kalman = Smart_CVM_Filter(0.016, 400, 0.25)
#kalman = Smart_CAM_Filter(0.016, 36, 0.25)
#pool1
#kalman = Smart_CVM_Filter(0.033, 500, 0.1)
#kalman = Smart_CAM_Filter(0.016, 820, 0.25)

# set cushion boundaries in kalman filter
kalman.setBoundaries(vh.xLeft, vh.xRight, vh.yTop, vh.yBot)

# instantiate ball detector (use specific parameters for video)
whiteBallDetection = BallDetection(*vh.giveParameters())

# frame counter
frame_no = 0

# array of filterd points
last_points_filtered = list()
# array of ground truth (ball detection without noise)
last_points = list()
# array of all predictions for every frame
filter_predictions = list()

# keep looping
while True:
    
    # get next frame
    frame = vh.giveFrames()

    # end if no frame available
    if frame is None:
        break
        
    # crop and resize
    frame = vh.cropFrame(frame)

    # detect ball in frame
    x,y,radius = whiteBallDetection.detectBall(frame)
    
    # draw circle around detection
    if (x is not None and y is not None):
        cv2.circle(frame, (int(x), int(y)), int(20), (255, 255, 255), 2)

    # add detected coordinates to array
    last_points.append([x, y])

    # draw line of the last detected points
    if len(last_points) > 1:
        last_point = None
        for point in last_points:
            if last_point is not None and point[0] is not None and last_point[0] is not None:
                cv2.line(frame, (int(last_point[0]),int(last_point[1])), (int(point[0]),int(point[1])), (0, 255, 255), 2)
            last_point = point

    # filter coordinates with kalman
    filterd = kalman.dofilter(x, y, radius)

    # draw line of the last filterd points
    last_points_filtered.append(filterd)
    if len(last_points_filtered) > 1:
        last_point = None
        for point in last_points_filtered:
            if last_point is not None:
                cv2.line(frame, (int(last_point[0]),int(last_point[1])), (int(point[0]),int(point[1])), (0, 255, 0), 2)
            last_point = point

    # get prediction for the next 60 frames      
    prePos, preVar = kalman.getPredictions(60)
    # draw predictions
    for i in range(0, len(prePos), 2):
        cv2.ellipse(frame, (prePos[i][0], prePos[i][1]), (int(1 * np.sqrt(preVar[i][0])), int(1 * np.sqrt(preVar[i][1]))), 0, 0, 360, (0, 200, 255), 2)

    # add predictions to array
    filter_predictions.append([])
    pre_pos, pre_var = kalman.getPredictions(max_count=61)
    filter_predictions[len(filter_predictions) - 1] = [pre_pos, pre_var]

    # draw curent filtered position    
    cv2.circle(frame, (int(filterd[0]), int(filterd[1])), 2, (255, 255, 0), 2)

    # draw cushions
    cv2.line(frame, p1, p2, (255, 255, 255), 2) #top bank
    cv2.line(frame, p2, p3, (255, 255, 255), 2) #right bank
    cv2.line(frame, p3, p4, (255, 255, 255), 2) #bottom bank
    cv2.line(frame, p4, p1, (255, 255, 255), 2) #left bank 
    
    # show the frame
    cv2.imshow("Frame", frame)
    cv2.waitKey(14)
    frame_no += 1

    
cv2.waitKey(1000)
cv2.destroyWindow("Frame")
cv2.waitKey(1000)
vh.vs.release()

print("MSE for prediction:")
print(get_mse_of_prediction(filter_predictions, last_points, 30))
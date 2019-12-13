import numpy as np
import cv2 

from pool_simulator import PoolSimulation
from filter.filter_constant_velocity import MyFilter

kalman = MyFilter(0.018, 300.0, 1.0)
sim = PoolSimulation(start_velocity = 900)

points = list()
filtered_points = list()

frame_no = 0
while frame_no < 1000:
    frame, position, velocity = sim.update()
    #cv2.line(frame, position, (position[0] + velocity[0], position[1] + velocity[1]), (255,0,0), 4)
    
    noised_position = (int(position[0] + np.random.randn() * 1), int(position[1] + np.random.randn() * 1))
    filtered = kalman.dofilter(noised_position[0], noised_position[1])

    last_point = None
    points.append(noised_position)
    if len(points) > 1:
        last_point = None
        for point in points:
            if last_point is not None:
                cv2.line(frame, (int(last_point[0]),int(last_point[1])), (int(point[0]),int(point[1])), (0,0,255), 2)
            last_point = point
    
    last_point = None
    filtered_points.append(filtered)
    if len(filtered_points) > 1:
        last_point = None
        for point in filtered_points:
            if last_point is not None:
                cv2.line(frame, (int(last_point[0]),int(last_point[1])), (int(point[0]),int(point[1])), (255,255,255), 2)
            last_point = point

    cv2.circle(frame, noised_position, 10, (0,0,255), -1)

    cv2.imshow("Pool Simulation", frame)
    frame_no += 1
    cv2.waitKey(16)
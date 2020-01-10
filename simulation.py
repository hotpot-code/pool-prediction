import numpy as np
import cv2
import math

from pool_simulator import PoolSimulation
from filter.filter_constant_velocity import MyFilter

noise = 10
update_time_in_ms = 16
process_noise = 950
dynamic_process_noise = 8000

kalman = MyFilter(update_time_in_ms / 1000, process_noise, noise * 1.1)
kalman_dynamic = MyFilter(update_time_in_ms / 1000, process_noise, noise * 1.1)
sim = PoolSimulation(start_velocity = 900, ms=update_time_in_ms)

points = list()
noised_points = list()
filtered_points = list()
dynamic_filtered_points = list()

frame_no = 0
while sim.isBallMoving:
    frame, position, velocity = sim.update()
    #cv2.line(frame, position, (position[0] + velocity[0], position[1] + velocity[1]), (255,0,0), 4)
    
    noised_position = (int(position[0] + np.random.randn() * noise), int(position[1] + np.random.randn() * noise))

    if sim.isBallNearBank:
        kalman_dynamic.process_noise = dynamic_process_noise
    else:
        kalman_dynamic.process_noise = process_noise

    filtered = kalman.dofilter(noised_position[0], noised_position[1])
    filtered_dynamic = kalman_dynamic.dofilter(noised_position[0], noised_position[1])

    last_point = None
    noised_points.append(noised_position)
    if len(noised_points) > 1:
        last_point = None
        for point in noised_points:
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

    last_point = None
    points.append(position)
    if len(points) > 1:
        last_point = None
        for point in points:
            if last_point is not None:
                cv2.line(frame, (int(last_point[0]), int(last_point[1])), (int(point[0]), int(point[1])),
                         (0, 255, 0), 2)
            last_point = point

    dynamic_filtered_points.append(filtered_dynamic)

    cv2.circle(frame, noised_position, 10, (0,0,255), -1)

    cv2.imshow("Pool Simulation", frame)
    frame_no += 1
    cv2.waitKey(1)


def residual(points, ground_truth):
    sum = 0
    for i in range(0, len(points) - 1):
        point = points[i]
        gt = ground_truth[i]
        distance =  math.sqrt((point[0] - gt[0])**2 + (point[1] - gt[1])**2)
        sum += distance
    return sum / len(points)

print("filter: ")
print(residual(filtered_points, points))
print("filter with dynamic process noise: ")
print(residual(dynamic_filtered_points, points))
print("no filter: ")
print(residual(noised_points, points))
print("truth: ")
print(residual(points, points))
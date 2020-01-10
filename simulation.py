import numpy as np
import cv2
import math

from pool_simulator import PoolSimulation
from filter.filter_constant_velocity import MyFilter

noise = 2.5
update_time_in_ms = 16
process_noise = 950
dynamic_process_noise = 8000
show_video = True
start_velocity = 1800

kalman = MyFilter(update_time_in_ms / 1000, process_noise, noise * 1.1)
kalman_dynamic = MyFilter(update_time_in_ms / 1000, process_noise, noise * 1.1)
sim = PoolSimulation(start_velocity = start_velocity, ms=update_time_in_ms)

points = list()
noised_points = list()
filtered_points = list()
dynamic_filtered_points = list()


def residual(points, ground_truth):
    sum = 0
    for i in range(0, len(points) - 1):
        point = points[i]
        gt = ground_truth[i]
        distance =  math.sqrt((point[0] - gt[0])**2 + (point[1] - gt[1])**2)
        sum += distance
    return sum / len(points)

def find_best_process_noise(noise,start_velocity, update_time_in_ms, min=0, max=5000):
    print("von " + str(min) + " bis " + str(max))
    middle = min + (max - min) / 2
    if middle > min + 1:
        sim = PoolSimulation(start_velocity = start_velocity, ms=update_time_in_ms)
        kalman_min = MyFilter(update_time_in_ms / 1000, min, noise * 1.1)
        kalman_max = MyFilter(update_time_in_ms / 1000, max, noise * 1.1)
        points = list()
        min_filtered_points = list()
        max_filtered_points = list()

        while sim.isBallMoving:
            frame, position, velocity = sim.update()
            noised_position = (int(position[0] + np.random.randn() * noise), int(position[1] + np.random.randn() * noise))
            min_filtered = kalman_min.dofilter(noised_position[0], noised_position[1])
            max_filtered = kalman_max.dofilter(noised_position[0], noised_position[1])
            min_filtered_points.append(min_filtered)
            max_filtered_points.append(max_filtered)
            points.append(position)

        min_value = residual(min_filtered_points, points)
        max_value = residual(max_filtered_points, points)
        if min_value > max_value:
            return find_best_process_noise(noise,start_velocity, update_time_in_ms, min=middle, max=max)
        else:
            return find_best_process_noise(noise,start_velocity, update_time_in_ms, min=min, max=middle)
    else:
        return min

def find_best_dynamic_process_noise(noise,start_velocity, update_time_in_ms, process_noise, min=1000, max=10000):
    print("von " + str(min) + " bis " + str(max))
    middle = min + (max - min) / 2
    if middle > min + 1:
        sim = PoolSimulation(start_velocity=start_velocity, ms=update_time_in_ms)
        kalman_min = MyFilter(update_time_in_ms / 1000, process_noise, noise * 1.1)
        kalman_max = MyFilter(update_time_in_ms / 1000, process_noise, noise * 1.1)
        points = list()
        min_filtered_points = list()
        max_filtered_points = list()

        while sim.isBallMoving:
            frame, position, velocity = sim.update()
            noised_position = (int(position[0] + np.random.randn() * noise), int(position[1] + np.random.randn() * noise))

            if sim.isBallNearBank:
                kalman_min.process_noise = min
                kalman_max.process_noise = max
            else:
                kalman_min.process_noise = process_noise
                kalman_max.process_noise = process_noise

            min_filtered = kalman_min.dofilter(noised_position[0], noised_position[1])
            max_filtered = kalman_max.dofilter(noised_position[0], noised_position[1])
            min_filtered_points.append(min_filtered)
            max_filtered_points.append(max_filtered)
            points.append(position)

        min_value = residual(min_filtered_points, points)
        max_value = residual(max_filtered_points, points)
        if min_value > max_value:
            return find_best_process_noise(noise, start_velocity, update_time_in_ms, min=middle, max=max)
        else:
            return find_best_process_noise(noise, start_velocity, update_time_in_ms, min=min, max=middle)
    else:
        return min

#process_noise = find_best_process_noise(noise, 900, update_time_in_ms, 500, 2000)

#dynamic_process_noise = find_best_dynamic_process_noise(noise, 900, update_time_in_ms, process_noise, 5000, 10000)

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

    noised_points.append(noised_position)
    filtered_points.append(filtered)
    points.append(position)
    dynamic_filtered_points.append(filtered_dynamic)

    if show_video:
        if len(noised_points) > 1:
            last_point = None
            for point in noised_points:
                if last_point is not None:
                    cv2.line(frame, (int(last_point[0]),int(last_point[1])), (int(point[0]),int(point[1])), (0,0,255), 2)
                last_point = point

        if len(filtered_points) > 1:
            last_point = None
            for point in filtered_points:
                if last_point is not None:
                    cv2.line(frame, (int(last_point[0]),int(last_point[1])), (int(point[0]),int(point[1])), (255,255,255), 2)
                last_point = point

        if len(points) > 1:
            last_point = None
            for point in points:
                if last_point is not None:
                    cv2.line(frame, (int(last_point[0]), int(last_point[1])), (int(point[0]), int(point[1])),
                             (0, 255, 0), 2)
                last_point = point

        cv2.circle(frame, noised_position, 10, (0,0,255), -1)
        cv2.imshow("Pool Simulation", frame)
        cv2.waitKey(update_time_in_ms)

    frame_no += 1




print("filter: ")
print(residual(filtered_points, points))
print("filter with dynamic process noise: ")
print(residual(dynamic_filtered_points, points))
print("no filter: ")
print(residual(noised_points, points))
print("truth: ")
print(residual(points, points))
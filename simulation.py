import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from pool_simulator import PoolSimulation
from filter.filter_constant_velocity import MyFilter

noise = 5
update_time_in_ms = 16
process_noise = 5000
dynamic_process_noise = 8000
show_video = True
start_velocity = 1400


def residual(points, ground_truth):
    sum_x = 0
    sum_y = 0
    for i in range(0, len(points) - 1):
        point = points[i]
        gt = ground_truth[i]
        distance_x = (point[0] - gt[0])**2
        distance_y = (point[1] - gt[1]) ** 2
        sum_x += distance_x
        sum_y += distance_y
    mse_x = sum_x / len(points)
    mse_y = sum_y / len(points)
    return mse_x + mse_y

def get_residual(noise,start_velocity, update_time_in_ms, process_noise):
    sim_test = PoolSimulation(start_velocity = start_velocity, ms=update_time_in_ms)
    kalman_test = MyFilter(update_time_in_ms / 1000, process_noise, noise)
    points = list()
    filtered_points = list()
    while sim_test.isBallMoving:
        frame, position, velocity = sim_test.update()
        R = np.diag([noise, noise]) ** 2
        noised_position = np.random.multivariate_normal(np.array(position).flatten(), R)
        filtered = kalman_test.dofilter(noised_position[0], noised_position[1])
        filtered_points.append(filtered)
        points.append(position)
    return residual(filtered_points, points)

def find_best_process_noise(noise,start_velocity, update_time_in_ms, min=0, max=5000):
    print("von " + str(min) + " bis " + str(max))
    middle = min + (max - min) / 2
    if middle > min + 1:
        sim = PoolSimulation(start_velocity = start_velocity, ms=update_time_in_ms)
        kalman_min = MyFilter(update_time_in_ms / 1000, min, noise)
        kalman_max = MyFilter(update_time_in_ms / 1000, max, noise)
        points = list()
        min_filtered_points = list()
        max_filtered_points = list()

        while sim.isBallMoving:
            frame, position, velocity = sim.update()
            R = np.diag([noise, noise]) ** 2
            noised_position = np.random.multivariate_normal(np.array(position).flatten(), R)
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
        kalman_min = MyFilter(update_time_in_ms / 1000, process_noise, noise)
        kalman_max = MyFilter(update_time_in_ms / 1000, process_noise, noise)
        points = list()
        min_filtered_points = list()
        max_filtered_points = list()

        while sim.isBallMoving:
            frame, position, velocity = sim.update()
            R = np.diag([noise, noise]) ** 2
            noised_position = np.random.multivariate_normal(np.array(position).flatten(), R)

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

noise_arr = list()
residual_arr = list()

min_pn = 0
min_residual = 1000000

pbar = trange(4000, 6000, 5)
for i in pbar:
    pbar.set_description("Processing with %s as process noise (current minimum residual: %s (%s))" % (i, min_residual, min_pn))
    cur_residual = get_residual(noise, start_velocity, update_time_in_ms, i)
    if cur_residual < min_residual:
        min_pn = i
        min_residual = cur_residual

process_noise = min_pn



#process_noise = find_best_process_noise(noise, start_velocity, update_time_in_ms, 500, 6000)
#print(process_noise)

#dynamic_process_noise = find_best_dynamic_process_noise(noise, 900, update_time_in_ms, process_noise, 5000, 10000)

kalman = MyFilter(update_time_in_ms / 1000, process_noise, noise)
kalman_dynamic = MyFilter(update_time_in_ms / 1000, process_noise, noise)
sim = PoolSimulation(start_velocity = start_velocity, ms=update_time_in_ms)

points = list()
noised_points = list()
filtered_points = list()
dynamic_filtered_points = list()

frame_no = 0
while sim.isBallMoving:
    frame, position, velocity = sim.update()
    #cv2.line(frame, position, (position[0] + velocity[0], position[1] + velocity[1]), (255,0,0), 4)
    R = np.diag([noise, noise]) ** 2
    noised_position = np.random.multivariate_normal(np.array(position).flatten(), R)
    #noised_position = (int(position[0] + np.random.randn() * noise), int(position[1] + np.random.randn() * noise))

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

        cv2.circle(frame, (int(noised_position[0]), int(noised_position[1])), 10, (0,0,255), -1)
        cv2.namedWindow('Pool Simulation', cv2.WINDOW_NORMAL)
        cv2.imshow("Pool Simulation", frame)
        cv2.resizeWindow('Pool Simulation', 800, 450)
        cv2.waitKey(1)

    frame_no += 1

print("filter: ")
print(10 * np.log10(residual(filtered_points, points)))
print("filter with dynamic process noise: ")
print(10 * np.log10(residual(dynamic_filtered_points, points)))
print("no filter: ")
print(10 * np.log10(residual(noised_points, points)))

# plt.plot(noise_arr, residual_arr, label='residual')
#
# plt.xlabel('process noise')
# plt.ylabel('residual')
#
# plt.title("Simple Plot")
#
# plt.legend()

plt.show()
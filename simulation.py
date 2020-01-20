import numpy as np
import cv2
import copy
import math
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

import time
current_milli_time = lambda: int(round(time.time() * 1000))

from pool_simulator import PoolSimulation
from filter.filter import MyFilter

noise = 6.1
update_time_in_secs = 0.016
process_noise = 2
dynamic_process_noise = 800
show_video = False
# https://billiards.colostate.edu/faq/speed/typical/
start_velocity = 660


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

kalman = MyFilter(update_time_in_secs, process_noise, noise)
kalman_dynamic = MyFilter(update_time_in_secs, process_noise, noise)
sim = PoolSimulation(start_angle = -0.7, start_velocity = start_velocity, seconds=update_time_in_secs, friction=10.3)

points = list()
noised_points = list()
filtered_points = list()
dynamic_filtered_points = list()
prediction_15 = list()
prediction_30 = list()
prediction_60 = list()
bank_time_span = list()

prePos_cached = np.array([])

bank_hits = 0
high_noise_mode = False

bank_start_frame = 0
near_bank = False

frame_no = 0
while sim.isBallMoving:
    start_ms = current_milli_time()
    frame, position, velocity = sim.update()
    #cv2.line(frame, position, (position[0] + velocity[0], position[1] + velocity[1]), (255,0,0), 4)
    R = np.diag([noise, noise]) ** 2
    noised_position = np.random.multivariate_normal(np.array(position).flatten(), R)
    #noised_position = (int(position[0] + np.random.randn() * noise), int(position[1] + np.random.randn() * noise))

    if sim.isBallNearBank:
        if not near_bank:
            bank_start_frame = frame_no
        near_bank = True
    else:
        if near_bank:
            bank_time_span.append((bank_start_frame, frame_no))
        near_bank = False

    if sim.isBallNearBank and bank_hits < sim.bank_hits:
        high_noise_mode = True
        bank_hits = sim.bank_hits
    if not sim.isBallNearBank and high_noise_mode:
        high_noise_mode = False

    if high_noise_mode:
        kalman_dynamic.setProcessNoise(dynamic_process_noise)
    else:
        kalman_dynamic.setProcessNoise(process_noise)

    filtered = kalman.dofilter(noised_position[0], noised_position[1])
    # skip frames
    if frame_no % 1 == 0:
        filtered_dynamic = kalman_dynamic.dofilter(noised_position[0], noised_position[1])
    else:
        filtered_dynamic = kalman_dynamic.dofilter(None, None)

    noised_points.append(noised_position)
    filtered_points.append(filtered)
    points.append(position)
    dynamic_filtered_points.append(filtered_dynamic)

    prePos, preVar = kalman_dynamic.getPredictions(max_var=60000, max_count=100)
    prediction_15.append(prePos[14])
    prediction_30.append(prePos[29])
    prediction_60.append(prePos[59])

    if show_video:
        if len(noised_points) > 1:
            last_point = None
            for point in noised_points:
                if last_point is not None:
                    cv2.line(frame, (int(last_point[0]),int(last_point[1])), (int(point[0]),int(point[1])), (0,0,255), 2)
                last_point = point

        if len(dynamic_filtered_points) > 1:
            last_point = None
            for point in dynamic_filtered_points:
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


        vel = np.array([kalman_dynamic.x_post[1, 0], kalman_dynamic.x_post[4, 0]])
        vel_norm = vel / np.linalg.norm(vel)
        cv2.arrowedLine(frame, (300, 300), (int(300 + vel_norm[0] * 100), int(300 + vel_norm[1] * 100)), (255, 255, 255))

        if not sim.isBallNearBank:
            prePos_cached = np.array(prePos)
            preVar_cached = np.array(preVar)
        else:
            if len(prePos_cached) > 0:
                prePos_cached = np.delete(prePos_cached, 0, axis=0)
                preVar_cached = np.delete(preVar_cached, 0, axis=0)

        for i in range(0, len(prePos_cached), 2):
            cv2.ellipse(frame, (prePos_cached[i][0], prePos_cached[i][1]), (int(4* np.sqrt(preVar_cached[i][0])), int(4*np.sqrt(preVar_cached[i][1]))), 0, 0, 360, (0, 200, 255), 2)
        
        if frame_no > prediction_offset:
            prediction_10_frames_behind = prediction_10[frame_no - prediction_offset]
            cv2.circle(frame, (int(prediction_10_frames_behind[0]), int(prediction_10_frames_behind[1])), 25, (100, 200, 80),20)

        cv2.namedWindow('Pool Simulation', cv2.WINDOW_NORMAL)
        cv2.imshow("Pool Simulation", frame)
        cv2.resizeWindow('Pool Simulation', 1200, 800)
        cv2.moveWindow('Pool Simulation', 0, 0)
        end_ms = current_milli_time()
        execution_time_in_ms = end_ms - start_ms
        cv2.waitKey(max(int(update_time_in_secs * 1000) - execution_time_in_ms, 1))
    frame_no += 1


cv2.waitKey()

print("filter: ")
print(10 * np.log10(residual(filtered_points, points)))
print("filter with dynamic process noise: ")
print(10 * np.log10(residual(dynamic_filtered_points, points)))
print("no filter: ")
print(10 * np.log10(residual(noised_points, points)))

prediction_15_residuals = list()
prediction_30_residuals = list()
prediction_60_residuals = list()

for i in range(0, len(points)):
    gt = points[i]
    if i >= 15:
        prediction = prediction_15[i - 15]
        distance = math.sqrt((prediction[0] - gt[0])**2 + (prediction[1] - gt[1])**2)
        prediction_15_residuals.append(distance)
    else:
        prediction_15_residuals.append(0)
    if i >= 30:
        prediction = prediction_30[i - 30]
        distance = math.sqrt((prediction[0] - gt[0])**2 + (prediction[1] - gt[1])**2)
        prediction_30_residuals.append(distance)
    else:
        prediction_30_residuals.append(0)
    if i >= 60:
        prediction = prediction_60[i - 60]
        distance = math.sqrt((prediction[0] - gt[0])**2 + (prediction[1] - gt[1])**2)
        prediction_60_residuals.append(distance)
    else:
        prediction_60_residuals.append(0)


# plt.plot(prediction_15_residuals, label='Prediction 15 frames ago')
# plt.plot(prediction_30_residuals, label='Prediction 30 frames ago')
# plt.plot(prediction_60_residuals, label='Prediction 60 frames ago')
plt.boxplot([prediction_15_residuals, prediction_30_residuals, prediction_60_residuals], showfliers=False)
# print(bank_time_span)
# for i in range(0, len(bank_time_span)):
#     plt.axvspan(bank_time_span[i][0], bank_time_span[i][1], color='red', alpha=0.1)

plt.xlabel('frame no')
plt.ylabel('distance to ground truth')

plt.title("Predictions")

plt.legend()

plt.show()
import numpy as np
import cv2
import copy
import math
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import sys

import time
current_milli_time = lambda: int(round(time.time() * 1000))

from pool_simulator import PoolSimulation
from filter.filter import MyFilter

class Simulation():

    def __init__(self, noise = 1.1, start_velocity = 660, update_time_in_secs = 0.016):
        self.noise = noise
        self.update_time_in_secs = update_time_in_secs
        # https://billiards.colostate.edu/faq/speed/typical/
        self.start_velocity = start_velocity

        self.show_video = False

        self.points = list()
        self.prediction_15 = list()
        self.prediction_30 = list()
        self.prediction_60 = list()
        self.bank_time_span = list()



    def residual(self, points, ground_truth):
        residuals = list()
        for i in range(0, len(points)):
            point = points[i]
            gt = ground_truth[i]
            residuals.append((point[0] - gt[0], point[1] - gt[1]))
        r = np.array(residuals)
        rx = r[:,0]
        ry = r[:,1]
        mse = (rx ** 2 + ry ** 2).mean()
        return mse

    def run(self, process_noise = 20, dynamic_process_noise = 800, show_video = False, save_prediction = True):
        kalman = MyFilter(self.update_time_in_secs, process_noise, self.noise)
        kalman_dynamic = MyFilter(self.update_time_in_secs, process_noise, self.noise)
        sim = PoolSimulation(start_angle = -0.7, start_velocity = self.start_velocity, seconds=self.update_time_in_secs, friction=10.3)

        self.points = list()
        noised_points = list()
        filtered_points = list()
        dynamic_filtered_points = list()

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
            R = np.diag([self.noise, self.noise]) ** 2
            noised_position = np.random.multivariate_normal(np.array(position).flatten(), R)
            #noised_position = (int(position[0] + np.random.randn() * noise), int(position[1] + np.random.randn() * noise))

            if sim.isBallNearBank:
                if not near_bank:
                    bank_start_frame = frame_no
                near_bank = True
            else:
                if near_bank:
                    self.bank_time_span.append((bank_start_frame, frame_no))
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
            self.points.append(position)
            dynamic_filtered_points.append(filtered_dynamic)

            if save_prediction:
                prePos, preVar = kalman_dynamic.getPredictions(max_count=60)
                self.prediction_15.append(prePos[14])
                self.prediction_30.append(prePos[29])
                self.prediction_60.append(prePos[59])

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

                if len(self.points) > 1:
                    last_point = None
                    for point in self.points:
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
                
                cv2.namedWindow('Pool Simulation', cv2.WINDOW_NORMAL)
                cv2.imshow("Pool Simulation", frame)
                cv2.resizeWindow('Pool Simulation', 1200, 800)
                cv2.moveWindow('Pool Simulation', 0, 0)
                end_ms = current_milli_time()
                execution_time_in_ms = end_ms - start_ms
                cv2.waitKey(max(int(self.update_time_in_secs * 1000) - execution_time_in_ms, 1))
            frame_no += 1
        
        return (10 * np.log10(self.residual(dynamic_filtered_points, self.points)), 10 * np.log10(self.residual(filtered_points, self.points)), 10 * np.log10(self.residual(noised_points, self.points)))

    def find_best_process_noise(self, process_noise_range = (0, 30), process_noise_step = 0.1, dynamic_process_noise_range = (500, 1200), dynamic_process_noise_step = 10):
        best_process_noise = 0
        best_dynamic_process_noise = 0
        best_db = 10000

        outer_range = np.arange(process_noise_range[0], process_noise_range[1], process_noise_step)
        inner_range = np.arange(dynamic_process_noise_range[0], dynamic_process_noise_range[1], dynamic_process_noise_step)

        with tqdm(total=(len(outer_range) * len(inner_range))) as pbar:
            for process_noise in outer_range:
                for dynamic_process_noise in inner_range:
                    pbar.set_description("Testing with pn=%d and dpn=%d! Current best: %fdB with pn=%d and dpn=%d" % (process_noise, dynamic_process_noise, best_db, best_process_noise, best_dynamic_process_noise))
                    dynamic_db, filter_db, no_filter_db = self.run(process_noise, dynamic_process_noise, save_prediction=False)
                    if dynamic_db < best_db:
                        best_db = dynamic_db
                        best_process_noise = process_noise
                        best_dynamic_process_noise = dynamic_process_noise
                    pbar.update(1)
        
        print("Found best process noise for simulation with noise=%f and start_velocity=%f! Best: %fdB with pn=%d and dpn=%d" % (self.noise, self.start_velocity, best_db, best_process_noise, best_dynamic_process_noise))
        return (best_process_noise, best_dynamic_process_noise)


    def show_plot(self):
        prediction_15_residuals = list()
        prediction_30_residuals = list()
        prediction_60_residuals = list()

        for i in range(0, len(self.points)):
            gt = self.points[i]
            if i >= 15:
                prediction = self.prediction_15[i - 15]
                distance = math.sqrt((prediction[0] - gt[0])**2 + (prediction[1] - gt[1])**2)
                prediction_15_residuals.append(distance)
            else:
                prediction_15_residuals.append(0)
            if i >= 30:
                prediction = self.prediction_30[i - 30]
                distance = math.sqrt((prediction[0] - gt[0])**2 + (prediction[1] - gt[1])**2)
                prediction_30_residuals.append(distance)
            else:
                prediction_30_residuals.append(0)
            if i >= 60:
                prediction = self.prediction_60[i - 60]
                distance = math.sqrt((prediction[0] - gt[0])**2 + (prediction[1] - gt[1])**2)
                prediction_60_residuals.append(distance)
            else:
                prediction_60_residuals.append(0)


        #plt.boxplot([prediction_15_residuals, prediction_30_residuals, prediction_60_residuals], showfliers=False)
        plt.plot(prediction_15_residuals, label='Prediction 15 frames ago')
        plt.plot(prediction_30_residuals, label='Prediction 30 frames ago')
        plt.plot(prediction_60_residuals, label='Prediction 60 frames ago')
        for i in range(0, len(self.bank_time_span)):
            plt.axvspan(self.bank_time_span[i][0], self.bank_time_span[i][1], color='red', alpha=0.1)

        plt.xlabel('frame no')
        plt.ylabel('distance to ground truth')

        plt.title("Predictions")

        plt.legend()

        plt.show()



testing_noise = float(sys.argv[1])
testing_start_velocity = float(sys.argv[2])
testing_fps = float(sys.argv[3])


print("Starting first Test with noise=%f and start_velocity=%f and fps=%d" % (testing_noise, testing_start_velocity, testing_fps))
sim = Simulation(noise=testing_noise, start_velocity=testing_start_velocity, update_time_in_secs=(1.0/testing_fps))
sim.find_best_process_noise()


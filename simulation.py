import numpy as np
import cv2
import copy
import math
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

import time
current_milli_time = lambda: int(round(time.time() * 1000))

from pool_simulator import PoolSimulation
from filter.smart_filter import Smart_CAM_Filter
from filter.filter_constant_acceleration import CAM_Filter

class Simulation():

    def __init__(self, noise = 1.1, start_velocity = 660, update_time_in_secs = 0.016):
        self.noise = noise
        self.update_time_in_secs = update_time_in_secs
        # https://billiards.colostate.edu/faq/speed/typical/
        self.start_velocity = start_velocity

        self.show_video = False

        self.points = list()
        self.mse_list = list()
        self.bank_time_span = list()
        self.filter_predictions = list()



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

    def run(self, filters, show_video = False, save_prediction = True, show_prediction=0):
        
        sim = PoolSimulation(start_angle = -0.7, start_velocity = self.start_velocity, seconds=self.update_time_in_secs, friction=10.3)

        self.names = list()
        for i in range(0, len(filters)):
            self.names.append(filters[i].name)

        self.points = list()
        noised_points = list()

        filter_points = list()
        for i in range(0, len(filters)):
            filter_points.append(list())

        self.filter_predictions = list()
        for i in range(0, len(filters)):
            self.filter_predictions.append(list())


        bank_hits = 0
        bank_start_frame = 0
        near_bank = False

        frame_no = 0
        while sim.isBallMoving:
            start_ms = current_milli_time()
            frame, position, velocity = sim.update()
            R = np.diag([self.noise, self.noise]) ** 2
            noised_position = np.random.multivariate_normal(np.array(position).flatten(), R)

            if sim.isBallNearBank:
                if not near_bank:
                    bank_start_frame = frame_no
                near_bank = True
            else:
                if near_bank:
                    self.bank_time_span.append((bank_start_frame, frame_no))
                near_bank = False

            for i, custom_filter in enumerate(filters):
                filter_points[i].append(custom_filter.dofilter(noised_position[0], noised_position[1]))

            noised_points.append(noised_position)
            self.points.append(position)

            if save_prediction:
                for i, custom_filter in enumerate(filters):
                    self.filter_predictions[i].append([])
                    pre_pos, pre_var = custom_filter.getPredictions(max_count=60)
                    self.filter_predictions[i][len(self.filter_predictions[i]) - 1] = [pre_pos, pre_var]

            if show_video:
                
                colors = [(255,255,0),(0,255,255), (255,0,255), (55,20,100)]
                for i, filter_point in enumerate(filter_points):
                    if len(filter_point) > 1:
                        last_point = None
                        for point in filter_point:
                            if last_point is not None:
                                cv2.line(frame, (int(last_point[0]),int(last_point[1])), (int(point[0]),int(point[1])), colors[i], 2)
                            last_point = point

                if len(self.points) > 1:
                    last_point = None
                    for point in self.points:
                        if last_point is not None:
                            cv2.line(frame, (int(last_point[0]), int(last_point[1])), (int(point[0]), int(point[1])),
                                    (0, 255, 0), 2)
                        last_point = point

                cv2.circle(frame, (int(noised_position[0]), int(noised_position[1])), 10, (0,0,255), -1)

                if show_prediction > -1 and show_prediction < len(self.filter_predictions):
                    prePos = self.filter_predictions[show_prediction][frame_no][0]
                    preVar = self.filter_predictions[show_prediction][frame_no][1]
                    for i in range(0, len(prePos)):
                        cv2.ellipse(frame, (prePos[i][0], prePos[i][1]), (int(4* np.sqrt(preVar[i][0])), int(4*np.sqrt(preVar[i][1]))), 0, 0, 360, (0, 200, 255), 2)

                cv2.namedWindow('Pool Simulation', cv2.WINDOW_NORMAL)
                cv2.imshow("Pool Simulation", frame)
                cv2.resizeWindow('Pool Simulation', 1200, 800)
                cv2.moveWindow('Pool Simulation', 0, 0)
                end_ms = current_milli_time()
                execution_time_in_ms = end_ms - start_ms
                cv2.waitKey(max(int(self.update_time_in_secs * 1000) - execution_time_in_ms, 1))
            frame_no += 1

        self.mse_list = list()
        for i in range(0, len(filter_points)):
            self.mse_list.append(10 * np.log10(self.residual(filter_points[i], self.points)))
        self.mse_list.append(10 * np.log10(self.residual(noised_points, self.points)))

        output_string = ""
        for i in range(0, len(filters)):
            output_string += "%s: %fdB " % (filters[i].name, self.mse_list[i])
        output_string += "No Filter: %fdB " % self.mse_list[-1]
        print(output_string)

        return self.mse_list

    def get_predictions(self, filter=0):
        return self.filter_predictions[filter]
    
    def get_mse_of_prediction(self, filter = 0, pre_no = 10, offset=30):
        predictions = self.get_predictions(filter)
        pre_pos = np.array(predictions)[offset:-pre_no, 0, pre_no]
        points = np.array(self.points)[offset+pre_no:]
        return 10 * np.log10(self.residual(pre_pos, points))
    
    def get_prediction_residuals(self, filter = 0, pre_no = 10, offset=30):
        predictions = self.get_predictions(filter)
        pre_pos = np.array(predictions)[offset:-pre_no-1, 0, pre_no-1]
        points = np.array(self.points)[offset+pre_no-1:]

        residuals = list()
        for i in range(0, len(pre_pos)):
            prediction = pre_pos[i]
            gt = points[i]
            distance = math.sqrt((prediction[0] - gt[0])**2 + (prediction[1] - gt[1])**2)
            residuals.append(distance)

        return residuals

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

    def show_mse_comparison_plot(self, pre_no=30):
        x = np.arange(len(self.mse_list))
        width = 0.35

        pre_mse_list = list()
        for i in range(0, len(self.mse_list) - 1):
            pre_mse_list.append(self.get_mse_of_prediction(i, pre_no=pre_no))
        pre_mse_list.append(0)

        plt.bar(x - width/2, self.mse_list, width, label='MSE of position')
        for i, v in enumerate(self.mse_list):
            plt.text(i - width/2, v + 0.5,  "{:10.2f}dB".format(v), color='blue', fontweight='bold', ha='center', va='bottom')
        plt.bar(x + width/2, pre_mse_list, width, label='MSE prediction %d frames ago' % pre_no)
        for i, v in enumerate(pre_mse_list):
            plt.text(i + width/2, v + 0.5,  "{:10.2f}dB".format(v), color='orange', fontweight='bold', ha='center', va='bottom')
        plt.ylabel('mse')
        plt.xticks(x, [*self.names, "no Filter"])
        plt.title("Filter Comparison")

        plt.legend()

        plt.show()

    def show_prediction_plot(self, filter=0, pre_nos=(15, 30, 60)):
        
        for pre_no in pre_nos:
            residuals = self.get_prediction_residuals(filter, pre_no)
            x = np.arange(0, len(residuals)) + pre_no
            plt.plot(x, residuals, label='Prediction %d frames ago' % pre_no)

        #plt.boxplot([prediction_15_residuals, prediction_30_residuals, prediction_60_residuals], showfliers=False)

        for i in range(0, len(self.bank_time_span)):
            plt.axvspan(self.bank_time_span[i][0], self.bank_time_span[i][1], color='red', alpha=0.1)

        plt.xlabel('frame no')
        plt.ylabel('distance to ground truth')

        plt.title("Predictions")

        plt.legend()

        plt.show()


testing_noise = [2.0, 5.0, 10.0]
testing_start_velocity = [700, 500, 300]
testing_fps = [60, 30, 10]

sim = Simulation(noise=2.0, start_velocity=500, update_time_in_secs=(1.0/60))

normal_cam = CAM_Filter(1.0/60, 5000, 2.0)
cam_dynamic = Smart_CAM_Filter(1.0/60, 4000, 2.0, name="Dynamic PN", dynamic_process_noise=10000, smart_prediction=False).setBoundaries(100, 1820, 100, 980).setRadius(25)
cam_smart = Smart_CAM_Filter(1.0/60, 700, 2.0, name="Smart Prediction", dynamic_process_noise=None, smart_prediction=True).setBoundaries(100, 1820, 100, 980).setRadius(25)
cam_dynamic_smart = Smart_CAM_Filter(1.0/60, 600, 2.0, name="Dynamic and smart", dynamic_process_noise=1000, smart_prediction=True).setBoundaries(100, 1820, 100, 980).setRadius(25)

filters = [normal_cam, cam_dynamic, cam_smart, cam_dynamic_smart]
sim.run(filters, show_video=False, show_prediction=2, save_prediction=True)
print(sim.get_mse_of_prediction(filter=0, pre_no = 10))
#print("CAM %fdB Dynmaic %fdB No Dynamic %fdB No Filter %fdB" % (cam_mse, dynamic_mse, filter_mse, no_filter_mse))
sim.show_mse_comparison_plot()
#sim.show_prediction_plot()
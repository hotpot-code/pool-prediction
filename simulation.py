import numpy as np
import cv2
import copy
import math
import matplotlib.pyplot as plt
import csv

import time
current_milli_time = lambda: int(round(time.time() * 1000))

from pool_simulator import PoolSimulation
from filter.smart_filter import Smart_CAM_Filter
from filter.smart_cvm_filter import Smart_CVM_Filter
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

    def create_csv(self, filename="sim.csv"):
        f = open(filename, 'w')

        simulation = PoolSimulation(start_angle=-0.7, start_velocity=self.start_velocity, seconds=self.update_time_in_secs, friction=10.3, noise=self.noise)
        with f:
            writer = csv.writer(f)
            writer.writerow(["POS_X", "POS_Y", "VEL_X", "VEL_Y", "SENSOR_X", "SENSOR_Y", "NEAR_BANK", "TS"])
            while simulation.isBallMoving:
                frame, position, velocity, sensor_pos = simulation.update()
                near_bank = simulation.isBallNearBank
                writer.writerow([*position, *velocity, *sensor_pos, near_bank, self.update_time_in_secs])

    @staticmethod
    def get_noise_from_csv(file):
        f = open(file, 'r')

        x_sum = 0
        y_sum = 0

        with f:
            reader = csv.reader(f)
            rows = [r for r in reader]
            for i in range(1, len(rows)):
                pos_x = float(rows[i][0])
                pos_y = float(rows[i][1])
                sensor_x = float(rows[i][4])
                sensor_y = float(rows[i][5])

                x_dis = math.pow(pos_x - sensor_x, 2)
                y_dis = math.pow(pos_y - sensor_y, 2)

                x_sum += x_dis
                y_sum += y_dis

        noise_x = math.sqrt(x_sum / (len(rows) - 1))
        noise_y = math.sqrt(y_sum / (len(rows) - 1))
        return (noise_x, noise_y)

    @staticmethod
    def get_update_time_from_csv(file):
        f = open(file, 'r')

        with f:
            reader = csv.reader(f)
            rows = [r for r in reader]
            ts = float(rows[1][7])
        return ts


    def run(self, filters, show_video = False, save_prediction = True, show_prediction=0, show_output=True, file=None):

        if file is None:
            sim = PoolSimulation(start_angle = -0.7, start_velocity = self.start_velocity, seconds=self.update_time_in_secs, friction=10.3, noise=self.noise)
        else:
            self.update_time_in_secs = Simulation.get_update_time_from_csv(file)
            frame, position, velocity, sensor_position, sim = PoolSimulation.update_from_csv(file, 0)

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
            if file is None:
                frame, position, velocity, sensor_position = sim.update()
            else:
                frame, position, velocity, sensor_position, sim = PoolSimulation.update_from_csv(file, frame_no)
            noised_position = sensor_position

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
                    pre_pos, pre_var = custom_filter.getPredictions(max_count=61)
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
                    for i in range(0, len(prePos), 5):
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

        if show_output:
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

        plt.xticks(x, [*self.names, "no Filter"])
        plt.ylabel('distance to ground truth')

        plt.ylim(top=650)
        plt.title("Predictions")
        plt.legend()
        plt.show()

    def show_prediction_boxplot(self, filter=0, pre_nos=(15, 30, 60)):

        residuals = list()
        for pre_no in pre_nos:
            residuals.append(self.get_prediction_residuals(filter, pre_no))

        plt.boxplot(residuals, showfliers=False)

        plt.xticks(range(1, len(residuals) + 1), pre_nos)
        plt.xlabel('prediction # frames ago')
        plt.ylabel('distance to ground truth')
        plt.ylim(top=530)
        plt.title("Predictions")
        plt.legend()
        plt.show()



if __name__ == "__main__":

    # testing_noise = [2.0, 5.0, 10.0]
    # testing_start_velocity = [700, 500, 300]
    # testing_fps = [60, 30, 10]
    #
    # for noise in testing_noise:
    #     for vel in testing_start_velocity:
    #         for fps in testing_fps:
    #             sim = Simulation(noise=noise, start_velocity=vel, update_time_in_secs=(1.0 / fps))
    #             sim.create_csv("simulations/sim_" + str(noise) + "_" + str(vel) + "_" + str(fps) + ".csv")

    sim = Simulation()

    normal_cam = CAM_Filter(1.0/60, 45290, 2.0, name="CAM Filter")
    normal_cvm = Smart_CVM_Filter(1.0 / 60, 2210, 2.0, name="CVM Filter", smart_prediction=False).setBoundaries(100, 1820, 100, 980).setRadius(25)
    smart_cvm = Smart_CVM_Filter(1.0 / 60, 533, 2.0, name="Smart CVM").setBoundaries(100, 1820, 100, 980).setRadius(25)
    cam_dynamic = Smart_CAM_Filter(1.0/60, 400, 2.0, name="dynamic CAM", dynamic_process_noise=40000, smart_prediction=False).setBoundaries(100, 1820, 100, 980).setRadius(25)
    cvm_dynamic = Smart_CVM_Filter(1.0/60, 300, 2.0, name="dynamic CVM", dynamic_process_noise=2210, smart_prediction=False).setBoundaries(100, 1820, 100, 980).setRadius(25)

    cam_smart = Smart_CAM_Filter(1.0/60, 511, 2.0, name="Smart CAM", dynamic_process_noise=None, smart_prediction=True).setBoundaries(100, 1820, 100, 980).setRadius(25)
    smart_dyn_cvm = Smart_CVM_Filter(1.0 / 60, 350, 2.0, name="dynamic smart CVM", dynamic_process_noise=860,).setBoundaries(100, 1820, 100, 980).setRadius(25)
    cam_dynamic_smart = Smart_CAM_Filter(1.0/60, 350, 2.0, name="dynamic smart CAM", dynamic_process_noise=860, smart_prediction=True).setBoundaries(100, 1820, 100, 980).setRadius(25)

    filters = [cvm_dynamic, cam_dynamic, smart_cvm]
    sim.run(filters, show_video=False, show_prediction=2, save_prediction=True, file="simulations/sim_2.0_700_60.csv")
    #sim.show_mse_comparison_plot(pre_no=30)
    sim.show_prediction_boxplot(filter=2,pre_nos=(15, 30, 60))
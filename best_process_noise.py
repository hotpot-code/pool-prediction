
import numpy as np
from tqdm import tqdm, trange
from filter.smart_filter import Smart_CAM_Filter
from simulation import Simulation
from multiprocessing import Process
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from filter.smart_cvm_filter import Smart_CVM_Filter

def find_best_process_noise(my_filter, file, process_noise_range = range(2000, 3000, 1)):
    best_process_noise = 0
    best_db = 10000
    mse = list()
    with tqdm(total=(len(process_noise_range))) as pbar:
        for process_noise in process_noise_range:
            sim = Simulation()
            my_filter.__init__(my_filter.Ts, process_noise=process_noise, sensor_noise=my_filter.sensor_noise, smart_prediction=my_filter.smart_prediction, dynamic_process_noise=my_filter.dynamic_process_noise)
            pbar.set_description("Testing with pn=%f! Current best: %fdB with pn=%f" % (my_filter.process_noise, best_db, best_process_noise))
            filter_db, no_filter_db = sim.run([my_filter], show_video=False, save_prediction=False, show_output=False, file=file)
            mse.append(filter_db)
            if filter_db < best_db:
                best_db = filter_db
                best_process_noise = process_noise
            pbar.update(1)

    print("Found best process noise for simulation with noise=%f and start_velocity=%f! Best: %fdB with pn=%f" % (my_filter.sensor_noise, sim.start_velocity, best_db, best_process_noise))
    return best_process_noise


def find_best_dynamic_process_noise(my_filter, file, dynamic_process_noise_range=range(8000, 15000, 100)):
    best_process_noise = 0
    best_db = 10000
    mse = list()
    with tqdm(total=(len(dynamic_process_noise_range))) as pbar:
        for dynamic_process_noise in dynamic_process_noise_range:
            sim = Simulation()
            my_filter.__init__(my_filter.Ts, my_filter.process_noise, my_filter.sensor_noise,
                               smart_prediction=my_filter.smart_prediction,
                               dynamic_process_noise=dynamic_process_noise)
            pbar.set_description("Testing with dpn=%f! Current best: %fdB with pn=%f" % (
            my_filter.dynamic_process_noise, best_db, best_process_noise))
            filter_db, no_filter_db = sim.run([my_filter], show_video=False, save_prediction=False, show_output=False,
                                              file=file)
            mse.append(filter_db)
            if filter_db < best_db:
                best_db = filter_db
                best_process_noise = dynamic_process_noise
            pbar.update(1)

    print("Found best process noise for simulation with noise=%f and start_velocity=%f! Best: %fdB with pn=%f" % (
    my_filter.sensor_noise, sim.start_velocity, best_db, best_process_noise))
    return best_process_noise


if __name__ == '__main__':
        my_filter = Smart_CVM_Filter(0.01666, 350, 2.0, smart_prediction=True, dynamic_process_noise=None).setBoundaries(100, 1820, 100, 980).setRadius(25)
        find_best_process_noise(my_filter, file="simulations/sim_2.0_500_60.csv", process_noise_range = range(530, 550, 1))


import numpy as np
from tqdm import tqdm, trange
from filter.smart_filter import Smart_CAM_Filter
from simulation import Simulation
from pool_simulator import PoolSimulation
from multiprocessing import Process
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from filter.smart_cvm_filter import Smart_CVM_Filter

def find_best_process_noise_and_dynamic(my_filter, file, process_noise_range = range(1, 5, 1), dynamic_process_noise_range = range(0, 12, 2)):
    
    results = np.empty([len(dynamic_process_noise_range), len(process_noise_range)])
    with tqdm(total=(len(dynamic_process_noise_range) *  len(process_noise_range))) as pbar:
        for u, process_noise in enumerate(process_noise_range):
            for v, dynamic_process_noise in enumerate(dynamic_process_noise_range):
                sim = Simulation()
                my_filter.__init__(my_filter.Ts, process_noise=process_noise, sensor_noise=my_filter.sensor_noise, smart_prediction=my_filter.smart_prediction, dynamic_process_noise=dynamic_process_noise)
                filter_db, no_filter_db = sim.run([my_filter], show_video=False, save_prediction=False, show_output=False, file=file)
                #pred_db = sim.get_mse_of_prediction(pre_no=30)
                results[v, u] = filter_db
                pbar.update(1)

    plt.imshow(results, origin='lower', cmap='ocean', vmax=(np.min(results)+1))
    plt.colorbar(label='MSE')
    plt.xlabel("Process Noise")
    plt.ylabel("Dynamic Process Noise")
    plt.xticks(range(0, len(process_noise_range)), list(process_noise_range))
    plt.yticks(range(0, len(dynamic_process_noise_range)), list(dynamic_process_noise_range))
    plt.show()

def find_best_process_noise(my_filter, file, process_noise_range = range(0, 10000, 1000)):
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

    step = process_noise_range[1] - process_noise_range[0]
    next_step = int(step / 10.0)
    if (next_step >= 1):
        return find_best_process_noise(my_filter, file, range(best_process_noise - step, best_process_noise + step, next_step))

    # plt.plot(process_noise_range, mse, label='linear')
    # plt.xlabel("process noise")
    # plt.ylabel("MSE")
    # plt.annotate('best process noise', xy=(best_process_noise, best_db),  xycoords='data',
    #         xytext=(0.5, 0.5), textcoords='axes fraction',
    #         arrowprops=dict(facecolor='black', shrink=0.05),
    #         horizontalalignment='right', verticalalignment='top',
    #         )
    # plt.show()

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
    boundaries = (PoolSimulation.inset, PoolSimulation.inset + PoolSimulation.table_width, PoolSimulation.inset, PoolSimulation.inset + PoolSimulation.table_height)
    ball_radius = 52

    my_filter = Smart_CVM_Filter(1.0 / 60, 350, 9.0, smart_prediction=True, dynamic_process_noise=None).setBoundaries(*boundaries).setRadius(ball_radius)
    my_dyn_filter = Smart_CVM_Filter(0.01666, 350, 2.0, smart_prediction=False, dynamic_process_noise=None).setBoundaries(100, 1820, 100, 980).setRadius(25)
    find_best_process_noise(my_filter, file="simulations/sim_9.0_900_60.csv", process_noise_range = range(0, 10000, 1000))
    #find_best_process_noise_and_dynamic(my_dyn_filter, file="simulations/sim_2.0_500_60.csv", process_noise_range = range(100, 700, 20), dynamic_process_noise_range = range(1500, 9500, 500))

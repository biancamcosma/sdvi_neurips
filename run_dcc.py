import os
from tqdm import tqdm
import time

rmh_strategy = "global"
prior = 9

times = []

for size in tqdm([2, 4, 8, 16, 32]):
    for run in [1, 2, 3]:
        experiment_name = "infinite_gmm_d"
        experiment_name += str(size)
        experiment_name += "_prior" + str(prior)
        experiment_name += "_" + rmh_strategy
        experiment_name += "_run" + str(run)
        experiment_name += "_dcc"
        
        command = "time python3 run_gmm_baselines.py inference_algo=dcc name=" + experiment_name
        command += " data_file=finite_gmm_data_n1000_mean_dim" + str(size) + ".npz"
        command += " validation_data_file=finite_gmm_data_n500_mean_dim" + str(size) + "_validation.npz"
        command += " dim=" + str(size)

        print(command)

        start_time = time.time()
        os.system(command)
        end_time = time.time()

        times.append(end_time - start_time)

print("Times:", times)

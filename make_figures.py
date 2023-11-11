import matplotlib.pyplot as plt
import pickle
import numpy as np

plt.style.use('ggplot')

dimensions = ["2", "4", "8", "16", "32"]

# Plot estimation for the number of clusters
x = np.arange(len(dimensions))

for prior in ["9"]:

    print("Prior", prior)
    to_plot = {}
    
    for rmh in ["local", "global"]:
        print("RMH strategy: ", rmh)
        results = []
        for dim in dimensions:
            print("Dimension", dim)
            avg_num = 0
            for run in ["1", "2", "3"]:
                # Compute path
                first_dir = "infinite_gmm_d" + dim + "_prior" + prior + "_" + rmh + "_run" + run + "_dcc/"
                second_dir = "data_file=finite_gmm_data_n1000_mean_dim"
                second_dir += dim + ".npz_dim=" + dim
                second_dir += "_inference_algo=dcc_validation_data_file=finite_gmm_data_n500_mean_dim"
                second_dir += dim + "_validation.npz/"
                third_dir = "seed=42"

                data_dir = first_dir + second_dir + third_dir + "/"

                with open("experiments/" + data_dir + "cluster_probs.pickle", "rb") as f:
                    cluster_probs = pickle.load(f)

                trace = max(zip(cluster_probs.values(), cluster_probs.keys()))[1]
                num_clusters = trace.count(",")
                avg_num += num_clusters
                print("Num clusters", num_clusters)

            avg_num /= 3
            results.append(avg_num)
        to_plot[rmh + " RMH"] = results

    # From: https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
    width = 0.25 
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in to_plot.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        multiplier += 1

    ax.set_ylabel('Average #clusters across runs')
    ax.set_xlabel('Dataset dimensionality')
    plt.title('Performance of DCC with a misspecified prior (Poisson rate = ' + prior + ')', fontsize=12)
    ax.set_xticks(x + width, dimensions)
    ax.legend(loc='upper left', ncols=2)
    plt.axhline(y = 5, color = 'black', linestyle = '--') 

    
    plt.savefig("prior" + prior + ".svg")
    plt.show()


#Runtimes for misspecified priors
l = [667.2554080486298, 696.4854323863983, 696.1319587230682, 846.5612518787384, 863.3096857070923, 863.2412106990814, 812.725394487381, 792.0368137359619, 776.7044498920441, 1125.896535873413, 1124.238362312317, 1199.8951318264008, 1122.8425426483154, 1109.3185076713562, 1078.2756779193878]
local_rmh_runtimes = [(l[i] + l[i+1] + l[i+2]) / 3 for i in [0, 3, 6, 9, 12]]
g = [805.8160140514374, 784.7251534461975, 815.9346673488617, 934.3851108551025, 873.3406863212585, 830.1599669456482, 711.3846244812012, 705.8743615150452, 713.6091415882111, 966.9554617404938, 1108.7523484230042, 1133.3350093364716, 1223.4920456409454, 1172.0125994682312, 1194.0689277648926]
global_rmh_runtimes = [(g[i] + g[i+1] + g[i+2]) / 3 for i in [0, 3, 6, 9, 12]]

f = plt.figure() 
plt.plot(x, local_rmh_runtimes, label="local RMH")
plt.plot(x, global_rmh_runtimes, label = "global RMH")
plt.xticks(x, dimensions)

plt.xlabel('Dataset dimensionality')
plt.ylabel('Runtime (in seconds)')

plt.title("Runtime of DCC with a misspecified prior (Poisson rate = 9)", fontsize=12)

f.set_figwidth(7) 
f.set_figheight(4) 
plt.legend()

plt.savefig('runtimes.svg')
plt.show()

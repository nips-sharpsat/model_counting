import pickle
import os, sys
import numpy as np
from numpy import genfromtxt
from os import listdir

dir_path = "./alldata/"


dataset2label = {"cell9_20_20":  "Cell(9,20,20)",

                 "cell35_128_110": "Cell(35,128,110)",
                 "cell35_192_128": "Cell(35,192,128)",
                 "cell35_256_210": "Cell(35,256,210)",
                 "cell35_348_280": "Cell(35,348,280)",

                 "cell49_128_110": "Cell(49,128,110)",

                 "grid_10_9":  "GridWrld(10,10)",
                 "grid_10_12": "GridWrld(10,12)",
                 "grid_10_14": "GridWrld(10,14)",
                 "grid_12_14": "GridWrld(12,14)"}

def policy2label(policy, ablation=False):
    ret = {"vanilla": "SharpSAT",
            "struct": "Learned", #"Graph",
            "timestruct": "Graph+Time",
            "timedecode": "Learned", #"Graph+Time+Activity",
            "time_activity": "Time+Activity",
            "timeonly": "Time",
            "random": "Random"}

    if (ablation):
        ret["struct"] = "Graph"
        ret["timedecode"] = "Graph+Time+Activity"

    return ret[policy]

def get_seq_lens(dataset, policies, wall_clock=False):
    max_seq_len = 0
    seq_lens = {}
    wall_clocks = {}
    max_units_len = 0
    heatmaps = {
        "actions": {},
        "units": {}
    }

    for policy in policies:
        ds_dir = f"{policy}_{dataset}"
        full_ds_path = os.path.join(dir_path, ds_dir)

        files = [os.path.join(full_ds_path, f) for f in os.listdir(full_ds_path)]
        files.sort(reverse=False)

        seq_lens[policy] = []
        wall_clocks[policy] = []

        for file in files:
            infile = open(file,'rb')
            new_dict = pickle.load(infile)
            infile.close()

            sequence = new_dict["actions"]

            # # For Rule 35
            # infile=open(file, "r")
            # sequence = np.array([int(line.rstrip('\n')) for line in infile])
            # infile.close()
            # # End of for Rule 35

            max_seq_len = max(max_seq_len, len(sequence))
            seq_lens[policy] += [len(sequence)]

            units = new_dict["units"]
            max_units_len = max(max_units_len, len(units))

            if(wall_clock):
                wall_clocks[policy] += [new_dict["time"]]

        print(f"Sequence length for {dataset2label[dataset]}-{policy2label(policy)} (max/avg): {max_seq_len:.1f}/{np.mean(seq_lens[policy]):.1f}")

    if (wall_clock):
        return {"steps": seq_lens, "times": wall_clocks}
    else:
        return {"steps": seq_lens}


def get_heatmaps(dataset, policies, num_files_to_consider = -1):
    space, time = [int(i) + 1 for i in dataset.split("_")[1:]]
    heatmaps = {
        "actions": {},
        "units": {}
    }

    for policy in policies:
        ds_dir = f"{policy}_{dataset}"

        full_ds_path = os.path.join(dir_path, ds_dir)

        files = [os.path.join(full_ds_path, f) for f in os.listdir(full_ds_path)]
        files.sort(reverse=False)

        heatmap_action = np.zeros([time, space])
        cnt_heatmap_action = np.full([time, space], 1e-10)
        heatmap_units  = np.zeros([time, space])
        unit_files_so_far, max_files_for_units = 0, 1
        for file in files[:num_files_to_consider]:
            infile = open(file,'rb')
            new_dict = pickle.load(infile)
            infile.close()

            sequence = new_dict["actions"]
            units    = new_dict["units"]

            #uniq
            _, idx = np.unique(sequence, return_index=True)
            sequence = sequence[np.sort(idx)]

            decision_lower = 0
            decision_upper = -1

            decision_upper = decision_upper if decision_upper > 0 else len(sequence)
            norm_const = decision_upper - decision_lower
            for pos_in_seq, lit in enumerate(sequence):
                if(decision_lower <= pos_in_seq <= decision_upper):
                    var = abs(int(lit))
                    row, col = var2grid(space, var)
                    heatmap_action[row][col] += 1 - (pos_in_seq - decision_lower)/norm_const
                    cnt_heatmap_action[row][col] += 1

            if unit_files_so_far < max_files_for_units:
                for unit in units:
                    var = abs(int(unit))
                    row, col = var2grid(space, var)
                    heatmap_units[row][col] += 1
                unit_files_so_far += 1

        # # Normalize
        # heatmap_action /= len(files)

        heatmaps["actions"][policy] = heatmap_action / cnt_heatmap_action
        heatmaps["units"][policy]   = heatmap_units

    return heatmaps



""" This function generates a 20x20 evolution grid of an Elementary Cellular Automate.
The first row is randomly generated and the rule is itereatively applied to it.
"""
def cell_grid(rule):
    rule = int(rule)
    size = 20
    rule_bin = [0]*8
    for enum, i in enumerate(str(bin(rule))[2:]):
        rule_bin[7-enum] = int(i)

    grid = np.zeros([size + 1, size + 1], dtype=int)
    grid[0] = np.random.randint(2, size=size + 1)

    for i in range(1, size):
        for j in range(size):
            prev = (grid[i-1][j-1] << 2) + (grid[i-1][j] << 1) + grid[i-1][j+1]
            if(rule_bin[prev-1] > 1): print("Err",rule_bin[prev-1])
            grid[i][j] = rule_bin[7 - prev]

    return grid


""" This function maps the cellular CNF variable to its location on the evolution grid.
"""
def var2grid(size, var):
    var = abs(var) -1
    col = var % size
    row = (var - col) / size
    return int(row), int(col)


## datasets:
# "cell49"
# "cell9"

# "pickle" # Cell(35,128,110)
# "cell35_192_128"
# "cell35_348_280"
# "huge"  # Cell(35,256,210)

# "GRID_HUGE"
# "grid_10_8"
# "grid_10_12"
# "grid_10_14"
# "grid_12_14"

## policies:
# "vanilla"
# "timedecode"
# "time_activity"
# "timeonly
# "struct"
# "random"


# datasets = ["pickle", "cell35_192_128", "cell35_348_280", "huge"]
# datasets = ["grid_10_9", "grid_10_12", "grid_10_14", "grid_12_14"]
# policies = ["vanilla", 'struct']#'timedecode']#"struct"]#, "random"]
# policies = ['timedecode', "vanilla"]

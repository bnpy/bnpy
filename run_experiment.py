import bnpy
import numpy as np
import util
import sys

# Hyperparameter settings
K = int(sys.argv[1]) if len(sys.argv) > 1 else 200
alloc_model = 'FiniteHMM'
obs_model = 'AutoRegGauss'
learn_alg = 'VB'
out_path = 'experiments/mocap124-K%d' % K
dataset_title = 'MoCap124'
L_vals = [1, 4, 16]
max_laps = 1000
kwargs = {
    'ECovMat': 'diagcovfirstdiff',
    'sF': 0.5,
    'VMat': 'same',
    'sV': 0.5,
    'MMat': 'eye'
}

# Load data
dataset = util.load_full_mocap_data()

# Run experiment
util.run_experiment(dataset, alloc_model, obs_model, learn_alg, K, out_path,
				    L_vals=L_vals, max_laps=max_laps, run_dense=True,
				    **kwargs)

import bnpy
import numpy as np
import util

alloc_model = 'FiniteHMM'
obs_model = 'AutoRegGauss'
learn_alg = 'VB'
out_path = 'experiments/mocap-newargs'
dataset_title = 'MoCap6'

dataset = util.load_mocap_data()

K = 20
kwargs = {
    'ECovMat': 'diagcovfirstdiff',
    'sF': 0.5,
    'VMat': 'same',
    'sV': 0.5,
    'MMat': 'eye'
}

# experiment = util.run_experiment(dataset, alloc_model, obs_model, learn_alg, K, out_path, **kwargs)
experiment = util.load_experiment(out_path)
util.plot_loss(experiment, dataset_title, ymax=2.45)
util.plot_hamming(experiment, dataset_title)

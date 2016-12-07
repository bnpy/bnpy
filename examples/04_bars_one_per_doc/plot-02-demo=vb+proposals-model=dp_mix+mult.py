"""
=========================================
02: Training DP mixture model with birth and merge proposals
=========================================

How to train a DP mixture of multinomials.
"""
import bnpy
import numpy as np
import os

from matplotlib import pylab
import seaborn as sns

FIG_SIZE = (3, 3)
SMALL_FIG_SIZE = (1.5, 1.5)
pylab.rcParams['figure.figsize'] = FIG_SIZE

###############################################################################
# Read dataset from file.

dataset_path = os.path.join(bnpy.DATASET_PATH, 'bars_one_per_doc')
dataset = bnpy.data.BagOfWordsData.read_npz(
    os.path.join(dataset_path, 'dataset.npz'))

###############################################################################
#
# Make a simple plot of the raw data
X_csr_DV = dataset.getSparseDocTypeCountMatrix()
bnpy.viz.BarsViz.show_square_images(
    X_csr_DV[:10].toarray(), vmin=0, vmax=5)
pylab.tight_layout()


###############################################################################
#
# Setup: Function to show bars from start to end of training run

def show_bars_over_time(
        task_output_path=None,
        query_laps=[0, 1, 2, 5, None],
        ncols=10):
    '''
    '''
    nrows = len(query_laps)
    fig_handle, ax_handles_RC = pylab.subplots(
        figsize=(SMALL_FIG_SIZE[0] * ncols, SMALL_FIG_SIZE[1] * nrows),
        nrows=nrows, ncols=ncols, sharex=True, sharey=True)
    for row_id, lap_val in enumerate(query_laps):
        cur_model, lap_val = bnpy.load_model_at_lap(task_output_path, lap_val)
        cur_topics_KV = cur_model.obsModel.getTopics()
        # Plot the current model
        cur_ax_list = ax_handles_RC[row_id].flatten().tolist()
        bnpy.viz.BarsViz.show_square_images(
            cur_topics_KV,
            vmin=0.0, vmax=0.1,
            ax_list=cur_ax_list)
        cur_ax_list[0].set_ylabel("lap: %d" % lap_val)
    pylab.tight_layout()

###############################################################################
# From K=2 initial clusters
# -------------------------
# 
# Using random initialization

initname = 'randomlikewang'
K = 2

K2_trained_model, K2_info_dict = bnpy.run(
    dataset, 'DPMixtureModel', 'Mult', 'memoVB',
    output_path='/tmp/bars_one_per_doc/trymoves-K=%d-initname=%s/' % (
        K, initname),
    nTask=1, nLap=50, convergeThr=0.0001, nBatch=1,
    K=K, initname=initname,
    gamma0=50.0, lam=0.1,
    moves='birth,merge,shuffle,delete',
    b_startLap=2,
    m_startLap=5,
    d_startLap=10)

show_bars_over_time(K2_info_dict['task_output_path'])



###############################################################################
# K=10 initial clusters
# ---------------------
# 
# Using random initialization

initname = 'randomlikewang'
K = 10

K10_trained_model, K10_info_dict = bnpy.run(
    dataset, 'DPMixtureModel', 'Mult', 'memoVB',
    output_path='/tmp/bars_one_per_doc/trymoves-K=%d-initname=%s/' % (
        K, initname),
    nTask=1, nLap=50, convergeThr=0.0001, nBatch=1,
    K=K, initname=initname,
    gamma0=50.0, lam=0.1,
    moves='birth,merge,shuffle,delete',
    b_startLap=2,
    m_startLap=5,
    d_startLap=10)

show_bars_over_time(K10_info_dict['task_output_path'])

pylab.show()
"""
===============================================================
05: Training for HDP Topic Model with birth and merge proposals
===============================================================


"""
import bnpy
import numpy as np
import os
import sys

from matplotlib import pylab
import seaborn as sns

FIG_SIZE = (3, 3)
SMALL_FIG_SIZE = (1,1)
pylab.rcParams['figure.figsize'] = FIG_SIZE

###############################################################################
# Read dataset from file.

dataset_path = os.path.join(bnpy.DATASET_PATH, 'bars_one_per_doc')
dataset = bnpy.data.BagOfWordsData.read_npz(
    os.path.join(dataset_path, 'dataset.npz'))

###############################################################################
#
# Set the local step algorithmic keyword args

local_step_kwargs = dict(
    # Perform at most this many iterations at each document
    nCoordAscentItersLP=100,
    # Stop local iters early when max change in doc-topic counts < this thr
    convThrLP=0.01,
    restartLP=0,
    doMemoizeLocalParams=0,
    )

merge_kwargs = dict(
    m_startLap=5,
    m_pair_ranking_procedure='total_size',
    m_pair_ranking_direction='descending',
    )

birth_kwargs = dict(
    b_startLap=2,
    b_stopLap=6,
    b_Kfresh=3, 
    b_nRefineSteps=5,
    )

###############################################################################
#
# Setup: Helper function to plot bars at each stage of training

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
            vmin=0.0, vmax=0.06,
            ax_list=cur_ax_list)
        cur_ax_list[0].set_ylabel("lap: %d" % lap_val)
    pylab.tight_layout()

###############################################################################
# Training from K=3 with births
# -----------------------------
# 
# Initialization: 3 topics, using random initial guess

trained_model, info_dict = bnpy.run(
    dataset, 'HDPTopicModel', 'Mult', 'memoVB',
    output_path='/tmp/bars_one_per_doc/trymoves-model=hdp+mult-K=3-moves=birth,merge,shuffle/',
    nLap=20, convergeThr=0.001, nBatch=1,
    K=3, initname='randomlikewang',
    alpha=0.5, lam=0.1,
    moves='birth,merge,shuffle',
    **dict(list(merge_kwargs.items()) +
           list(local_step_kwargs.items()) + 
           list(birth_kwargs.items())))

show_bars_over_time(info_dict['task_output_path'])


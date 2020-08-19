"""
=======================================================================
Variational with birth and merge proposals for DP mixtures of Gaussians
=======================================================================

How to train a DP mixture model.

We'll show that despite diverse, poor quality initializations, our proposal moves that insert new clusters (birth) and remove redundant clusters (merge) can consistently recover the same ideal posterior with 8 clusters.

"""
# SPECIFY WHICH PLOT CREATED BY THIS SCRIPT IS THE THUMBNAIL IMAGE
# sphinx_gallery_thumbnail_number = 2

import bnpy
import numpy as np
import os

from matplotlib import pylab
import seaborn as sns

FIG_SIZE = (3, 3)
pylab.rcParams['figure.figsize'] = FIG_SIZE

###############################################################################
# Read dataset from file.

dataset_path = os.path.join(bnpy.DATASET_PATH, 'AsteriskK8')
dataset = bnpy.data.XData.read_npz(
    os.path.join(dataset_path, 'x_dataset.npz'))

###############################################################################
#
# Make a simple plot of the raw data

pylab.plot(dataset.X[:, 0], dataset.X[:, 1], 'k.')
pylab.gca().set_xlim([-2, 2])
pylab.gca().set_ylim([-2, 2])
pylab.tight_layout()

###############################################################################
#
# Setup: Function for visualization
# ---------------------------------
# Here's a short function to show the learned clusters over time.

def show_clusters_over_time(
        task_output_path=None,
        query_laps=[0, 1, 2, 5, 10, None],
        nrows=2):
    ''' Read model snapshots from provided folder and make visualizations

    Post Condition
    --------------
    New matplotlib plot with some nice pictures.
    '''
    ncols = int(np.ceil(len(query_laps) // float(nrows)))
    fig_handle, ax_handle_list = pylab.subplots(
        figsize=(FIG_SIZE[0] * ncols, FIG_SIZE[1] * nrows),
        nrows=nrows, ncols=ncols, sharex=True, sharey=True)
    for plot_id, lap_val in enumerate(query_laps):
        cur_model, lap_val = bnpy.load_model_at_lap(task_output_path, lap_val)
        # Plot the current model
        cur_ax_handle = ax_handle_list.flatten()[plot_id]
        bnpy.viz.PlotComps.plotCompsFromHModel(
            cur_model, Data=dataset, ax_handle=cur_ax_handle)
        cur_ax_handle.set_xticks([-2, -1, 0, 1, 2])
        cur_ax_handle.set_yticks([-2, -1, 0, 1, 2])
        cur_ax_handle.set_xlabel("lap: %d" % lap_val)
    pylab.tight_layout()


###############################################################################
# Training from K=1 cluster
# -------------------------
# 
# Using 1 initial cluster, with birth and merge proposal moves.

K1_trained_model, K1_info_dict = bnpy.run(
    dataset, 'DPMixtureModel', 'Gauss', 'memoVB',
    output_path='/tmp/AsteriskK8/trymoves-K=1/',
    nLap=100, nTask=1, nBatch=1,
    sF=0.1, ECovMat='eye',
    K=1, initname='randexamples',
    moves='birth,merge,shuffle',
    m_startLap=5, b_startLap=2, b_Kfresh=4)

show_clusters_over_time(K1_info_dict['task_output_path'])

###############################################################################
# Training from K=4 cluster
# -------------------------
# 
# Now using 4 initial clusters, with birth and merge proposal moves.

K4_trained_model, K4_info_dict = bnpy.run(
    dataset, 'DPMixtureModel', 'Gauss', 'memoVB',
    output_path='/tmp/AsteriskK8/trymoves-K=4/',
    nLap=100, nTask=1, nBatch=1,
    sF=0.1, ECovMat='eye',
    K=4, initname='randexamples',
    moves='birth,merge,shuffle',
    m_startLap=5, b_startLap=2, b_Kfresh=4)

show_clusters_over_time(K4_info_dict['task_output_path'])

###############################################################################
# Training from K=8 cluster
# -------------------------
# 
# Now using 8 initial clusters

K8_trained_model, K8_info_dict = bnpy.run(
    dataset, 'DPMixtureModel', 'Gauss', 'memoVB',
    output_path='/tmp/AsteriskK8/trymoves-K=8/',
    nLap=100, nTask=1, nBatch=1,
    sF=0.1, ECovMat='eye',
    K=8, initname='randexamples',
    moves='birth,merge,shuffle',
    m_startLap=5, b_startLap=2, b_Kfresh=4)

show_clusters_over_time(K8_info_dict['task_output_path'])

###############################################################################
# Training from K=25 cluster
# --------------------------
# 
# Now using 25 initial clusters

K25_trained_model, K25_info_dict = bnpy.run(
    dataset, 'DPMixtureModel', 'Gauss', 'memoVB',
    output_path='/tmp/AsteriskK8/trymoves-K=25/',
    nLap=100, nTask=1, nBatch=1,
    sF=0.1, ECovMat='eye',
    K=25, initname='randexamples',
    moves='birth,merge,shuffle',
    m_startLap=5, b_startLap=2, b_Kfresh=4)

show_clusters_over_time(K25_info_dict['task_output_path'])

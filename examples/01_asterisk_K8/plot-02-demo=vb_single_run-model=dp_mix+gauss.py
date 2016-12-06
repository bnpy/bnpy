"""
============================================================
Variational coordinate descent for Mixture of Gaussians
============================================================

How to do Variational Bayes (VB) coordinate descent for GMM.

Here, we train a finite mixture of Gaussians with full covariances.

We'll consider a mixture model with a symmetric Dirichlet prior:

.. math::

    \pi \sim \mbox{Dir}(1/K, 1/K, \ldots 1/K)

as well as a standard conjugate prior on the mean and covariances, such that

.. math::

    \E[\mu_k] = 0

    \E[\Sigma_k] = 0.1 I_D

We will initialize the approximate variational posterior 
using K=10 randomly chosen examples ('randexamples' procedure),
and then perform coordinate descent updates
(alternating local step and global step) until convergence.
"""
import bnpy
import numpy as np
import os

from matplotlib import pylab
import seaborn as sns
# sphinx_gallery_thumbnail_number = 3

FIG_SIZE = (3, 3)
pylab.rcParams['figure.figsize'] = FIG_SIZE

###############################################################################
# Read bnpy's built-in "AsteriskK8" dataset from file.

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
# Training the model
# ------------------
# Let's do one single run of the VB algorithm.
# 
# Using 10 clusters and the 'randexamples' initializatio procedure.

trained_model, info_dict = bnpy.run(
    dataset, 'FiniteMixtureModel', 'Gauss', 'VB',
    output_path='/tmp/AsteriskK8/helloworld-K=10/',
    nLap=100,
    sF=0.1, ECovMat='eye',
    K=10,
    initname='randexamples')

###############################################################################
# 
# Loss function trace plot
# ------------------------
# We can plot the value of the loss function over iterations,
# starting after the first full pass over the dataset (first lap).
#
# As expected, we see monotonic decrease in the loss function's score 
# after every subsequent iteration.
#
# Remember that the VB algorithm for GMMs is *guaranteed*
# to decrease this loss function after every step.
#
pylab.plot(info_dict['lap_history'][1:], info_dict['loss_history'][1:], 'k.-')
pylab.xlabel('num. laps')
pylab.ylabel('loss')
pylab.tight_layout()


###############################################################################
#
# Visualization of learned clusters
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
#
# Show the estimated clusters over time
show_clusters_over_time(info_dict['task_output_path'])
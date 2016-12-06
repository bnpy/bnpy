"""
========================================================================
Variational with merge and delete proposals for DP mixtures of Gaussians
========================================================================

How delete moves can be more effective than merges.

In this example, we show how merge moves alone may not be enough 
to reliably escape local optima. Instead, we show that more flexible
delete moves can escape from situations where merges alone fail.
"""
import bnpy
import numpy as np
import os

from matplotlib import pylab
import seaborn as sns

FIG_SIZE = (3, 3)
pylab.rcParams['figure.figsize'] = FIG_SIZE

###############################################################################
#
# Create toy dataset of many points drawn from standard normal

prng = np.random.RandomState(42)
X = prng.randn(100000, 1)
dataset = bnpy.data.XData(X, name='StandardNormalK1')

###############################################################################
#
# Make a simple plot of the raw data

pylab.hist(dataset.X[:, 0], 50, normed=1)
pylab.xlabel('x')
pylab.ylabel('p(x)')
pylab.tight_layout()


###############################################################################
# Setup: Determine specific settings of the proposals
# ---------------------------------------------------

merge_kwargs = dict(
    m_startLap=10,
    m_pair_ranking_procedure='total_size',
    m_pair_ranking_direction='descending',
    )

delete_kwargs = dict(
    d_startLap=10,
    d_nRefineSteps=50,
    )


###############################################################################
#
# Setup: Helper function to display the learned clusters
# ------------------------------------------------------

def show_clusters_over_time(
        task_output_path=None,
        query_laps=[0, 1, 2, 10, 20, None],
        nrows=2):
    '''
    '''
    ncols = int(np.ceil(len(query_laps) // float(nrows)))
    fig_handle, ax_handle_list = pylab.subplots(
        figsize=(FIG_SIZE[0] * ncols, FIG_SIZE[1] * nrows),
        nrows=nrows, ncols=ncols, sharex=True, sharey=True)
    for plot_id, lap_val in enumerate(query_laps):
        cur_model, lap_val = bnpy.load_model_at_lap(task_output_path, lap_val)
        cur_ax_handle = ax_handle_list.flatten()[plot_id]
        bnpy.viz.PlotComps.plotCompsFromHModel(
            cur_model, dataset=dataset, ax_handle=cur_ax_handle)
        cur_ax_handle.set_xlim([-4.5, 4.5])
        cur_ax_handle.set_xlabel("lap: %d" % lap_val)
    pylab.tight_layout()


###############################################################################
#
# Run with *merge* moves only, from K=5 initial clusters
# --------------------------------------------------------
#
# Unfortunately, no pairwise merge is accepted.
# The model is stuck using 5 clusters when one cluster would do.

gamma = 5.0
sF = 0.1
K = 5

m_trained_model, m_info_dict = bnpy.run(
    dataset, 'DPMixtureModel', 'Gauss', 'memoVB',
    output_path=('/tmp/StandardNormalK1/' + 
        'trymoves-K=%d-gamma=%s-ECovMat=%s*eye-moves=merge,shuffle/' % (
            K, gamma, sF)),
    nLap=100, nTask=1, nBatch=1,
    gamma0=gamma, sF=sF, ECovMat='eye',
    K=K, initname='randexamplesbydist',
    moves='merge,shuffle',
    **dict(**merge_kwargs))

show_clusters_over_time(m_info_dict['task_output_path'])

###############################################################################
#
# Run with *delete* moves, from K=5 initial clusters
# --------------------------------------------------------
#
# More flexible delete moves *are* accepted.

d_trained_model, d_info_dict = bnpy.run(
    dataset, 'DPMixtureModel', 'Gauss', 'memoVB',
    output_path=('/tmp/StandardNormalK1/' + 
        'trymoves-K=%d-gamma=%s-ECovMat=%s*eye-moves=delete,shuffle/' % (
            K, gamma, sF)),
    nLap=100, nTask=1, nBatch=1,
    gamma0=gamma, sF=sF, ECovMat='eye',
    K=K, initname='randexamplesbydist',
    moves='delete,shuffle',
    **dict(delete_kwargs))

show_clusters_over_time(d_info_dict['task_output_path'])


###############################################################################
# 
# Loss function trace plot
# ------------------------
#
pylab.plot(
    m_info_dict['lap_history'][1:],
    m_info_dict['loss_history'][1:], 'k.-',
    label='vb_with_merges')
pylab.plot(
    d_info_dict['lap_history'][1:],
    d_info_dict['loss_history'][1:], 'b.-',
    label='vb_with_deletes')
pylab.legend(loc='upper right')
pylab.xlabel('num. laps')
pylab.ylabel('loss')
pylab.tight_layout()


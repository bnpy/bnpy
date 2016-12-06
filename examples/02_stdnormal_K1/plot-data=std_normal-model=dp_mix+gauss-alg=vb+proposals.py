"""
============================================
Mixture of Gaussians: VB with proposal moves
============================================


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
# Create dataset of many points drawn from standard normal

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



merge_kwargs = dict(
    m_pair_ranking_procedure='total_size',
    m_pair_ranking_direction='descending',
    )

delete_kwargs = dict(
    d_nRefineSteps=50,
    )

###############################################################################
#
# Let's run VB coordinate descent algorithm
# with merge proposals to escape local optima
#
# Using 5 initial cluster.

gamma = 5.0
sF = 0.1

for K in [5]:

    trained_model, info_dict = bnpy.run(
        dataset, 'DPMixtureModel', 'Gauss', 'memoVB',
        output_path=('/tmp/StandardNormalK1/' + 
            'trymoves-K=%d-gamma=%s-ECovMat=%s*eye-moves=merge,shuffle/' % (
                K, gamma, sF)),
        nLap=100, nTask=1, nBatch=1,
        gamma0=gamma, sF=sF, ECovMat='eye',
        K=K, initname='randexamplesbydist',
        moves='merge,shuffle',
        m_startLap=10,
        **dict(**merge_kwargs))

    trained_model, info_dict = bnpy.run(
        dataset, 'DPMixtureModel', 'Gauss', 'memoVB',
        output_path=('/tmp/StandardNormalK1/' + 
            'trymoves-K=%d-gamma=%s-ECovMat=%s*eye-moves=delete,shuffle/' % (
                K, gamma, sF)),
        nLap=100, nTask=1, nBatch=1,
        gamma0=gamma, sF=sF, ECovMat='eye',
        K=K, initname='randexamplesbydist',
        moves='delete,shuffle',
        d_startLap=10,
        **dict(delete_kwargs))

###############################################################################
#
# Helper function to display the learned clusters
# as we've done more iterations (passes through the dataset)

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
# Show the clusters over time
task_output_path = '/tmp/StandardNormalK1/' + \
    'trymoves-K=5-gamma=%s-ECovMat=0.1*eye-moves=merge,shuffle/1/' % (
        gamma)
show_clusters_over_time(task_output_path)

###############################################################################
#
# Show the clusters over time
task_output_path = '/tmp/StandardNormalK1/' + \
    'trymoves-K=5-gamma=%s-ECovMat=0.1*eye-moves=delete,shuffle/1/' % (
        gamma)
show_clusters_over_time(task_output_path)

#pylab.show()
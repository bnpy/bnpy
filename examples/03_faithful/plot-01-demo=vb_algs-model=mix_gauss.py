"""
==============================================
Variational training for Mixtures of Gaussians
==============================================

Showcase of different models and algorithms applied to same dataset.

In this example, we show how bnpy makes it easy to apply
different models and algorithms to the same dataset.

"""
import bnpy
import numpy as np
import os

from matplotlib import pylab
import seaborn as sns

SMALL_FIG_SIZE = (2.5, 2.5)
FIG_SIZE = (5, 5)
pylab.rcParams['figure.figsize'] = FIG_SIZE

np.set_printoptions(precision=3, suppress=1, linewidth=200)

###############################################################################
#
# Load dataset from file

dataset_path = os.path.join(bnpy.DATASET_PATH, 'faithful')
dataset = bnpy.data.XData.read_csv(
    os.path.join(dataset_path, 'faithful.csv'))

###############################################################################
#
# Make a simple plot of the raw data

pylab.plot(dataset.X[:, 0], dataset.X[:, 1], 'k.')
pylab.xlabel(dataset.column_names[0])
pylab.ylabel(dataset.column_names[1])
pylab.tight_layout()
data_ax_h = pylab.gca()


###############################################################################
#
# Setup: Helper function to display the learned clusters
# ------------------------------------------------------

def show_clusters_over_time(
        task_output_path=None,
        query_laps=[0, 1, 2, 10, 20, None],
        nrows=2):
    ''' Show 2D elliptical contours overlaid on raw data.
    '''
    ncols = int(np.ceil(len(query_laps) // float(nrows)))
    fig_handle, ax_handle_list = pylab.subplots(
        figsize=(SMALL_FIG_SIZE[0] * ncols, SMALL_FIG_SIZE[1] * nrows),
        nrows=nrows, ncols=ncols, sharex=True, sharey=True)
    for plot_id, lap_val in enumerate(query_laps):
        cur_model, lap_val = bnpy.load_model_at_lap(task_output_path, lap_val)
        cur_ax_handle = ax_handle_list.flatten()[plot_id]
        bnpy.viz.PlotComps.plotCompsFromHModel(
            cur_model, dataset=dataset, ax_handle=cur_ax_handle)
        cur_ax_handle.set_title("lap: %d" % lap_val)
        cur_ax_handle.set_xlabel(dataset.column_names[0])
        cur_ax_handle.set_ylabel(dataset.column_names[1])
        cur_ax_handle.set_xlim(data_ax_h.get_xlim())
        cur_ax_handle.set_ylim(data_ax_h.get_ylim())
    pylab.tight_layout()


###############################################################################
#
# *DiagGauss* observation model
# -----------------------------
#
# Assume diagonal covariances.
#
# Start with too many clusters (K=20)

gamma = 5.0
sF = 5.0
K = 20

diag_trained_model, diag_info_dict = bnpy.run(
    dataset, 'DPMixtureModel', 'DiagGauss', 'memoVB',
    output_path='/tmp/faithful/showcase-K=20-lik=DiagGauss-ECovMat=5*eye/',
    nLap=1000, nTask=1, nBatch=1, convergeThr=0.0001,
    gamma0=gamma, sF=sF, ECovMat='eye',
    K=K, initname='randexamples',
    )
show_clusters_over_time(diag_info_dict['task_output_path'])

###############################################################################
#
# *Gauss* observations + VB
# -------------------------
#
# Assume full covariances.
#
# Start with too many clusters (K=20)

full_trained_model, full_info_dict = bnpy.run(
    dataset, 'DPMixtureModel', 'Gauss', 'VB',
    output_path='/tmp/faithful/showcase-K=20-lik=Gauss-ECovMat=5*eye/',
    nLap=1000, nTask=1, nBatch=1, convergeThr=0.0001,
    gamma0=gamma, sF=sF, ECovMat='eye',
    K=K, initname='randexamples',
    )
show_clusters_over_time(full_info_dict['task_output_path'])

###############################################################################
#
# *ZeroMeanGauss* observations + VB
# ---------------------------------
#
# Assume full covariances and fix all means to zero.
#
# Start with too many clusters (K=20)

zm_trained_model, zm_info_dict = bnpy.run(
    dataset, 'DPMixtureModel', 'ZeroMeanGauss', 'VB',
    output_path='/tmp/faithful/showcase-K=20-lik=ZeroMeanGauss-ECovMat=5*eye/',
    nLap=1000, nTask=1, nBatch=1, convergeThr=0.0001,
    gamma0=gamma, sF=sF, ECovMat='eye',
    K=K, initname='randexamples',
    )
show_clusters_over_time(zm_info_dict['task_output_path'])


###############################################################################
#
# *Gauss* observations + stochastic VB
# ------------------------------------
#
# Assume full covariances and fix all means to zero.
#
# Start with too many clusters (K=20)

stoch_trained_model, stoch_info_dict = bnpy.run(
    dataset, 'DPMixtureModel', 'Gauss', 'soVB',
    output_path=\
        '/tmp/faithful/showcase-K=20-lik=Gauss-ECovMat=5*eye-alg=soVB/',
    nLap=50, nTask=1, nBatch=50,
    rhoexp=0.51, rhodelay=1.0,
    gamma0=gamma, sF=sF, ECovMat='eye',
    K=K, initname='randexamples',
    )
show_clusters_over_time(stoch_info_dict['task_output_path'])


###############################################################################
# 
# Compare loss function traces for all methods
# --------------------------------------------
#
pylab.figure()

pylab.plot(
    zm_info_dict['lap_history'],
    zm_info_dict['loss_history'], 'b.-',
    label='full_covar zero_mean')
pylab.plot(
    full_info_dict['lap_history'],
    full_info_dict['loss_history'], 'k.-',
    label='full_covar')
pylab.plot(
    diag_info_dict['lap_history'],
    diag_info_dict['loss_history'], 'r.-',
    label='diag_covar')
pylab.plot(
    stoch_info_dict['lap_history'],
    stoch_info_dict['loss_history'], 'c.:',
    label='full_covar stochastic')
pylab.legend(loc='upper right')
pylab.xlabel('num. laps')
pylab.ylabel('loss')
pylab.xlim([4, 100]) # avoid early iterations
pylab.ylim([2.34, 4.0]) # handpicked
pylab.draw()
pylab.tight_layout()

###############################################################################
# 
# Inspect the learned distribution over appearance probabilities
# --------------------------------------------------------------

# E_proba_K : 1D array, size n_clusters
# Each entry gives expected probability of that cluster

E_proba_K = stoch_trained_model.allocModel.get_active_comp_probs()

print("probability of each cluster:")
print(E_proba_K)

###############################################################################
# 
# Inspect the learned means and covariance distributions
# ------------------------------------------------------
#
# Remember that each cluster has the following approximate posterior
# over its mean vector $\mu$ and covariance matrix $\Sigma$:
#
# $$
# q(\mu, \Sigma) = \Normal(\mu | m, \kappa \Sigma) \Wishart(\Sigma | \nu, S)
# $$
# 
# We show here how to compute the expected mean of $\mu$ and $\Sigma$
# from a trained model.

for k in range(K):
    E_mu_k = stoch_trained_model.obsModel.get_mean_for_comp(k)
    E_Sigma_k = stoch_trained_model.obsModel.get_covar_mat_for_comp(k)
    print("")
    print("mean[k=%d]" % k)
    print(E_mu_k)
    print("covar[k=%d]" % k)
    print(E_Sigma_k)


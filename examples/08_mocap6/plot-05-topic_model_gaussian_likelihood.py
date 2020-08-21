"""
======================================
Topic models with Gaussian likelihoods
======================================

Quick demonstration that you can easily use bnpy to
perform mixed membership modeling of grouped data with *any* likelihood.

The basic idea is that we use the same Gaussian mixture model for each
group of data, but the *appearance probabilities* are allowed to be 
learned in a customized way for each group.

Here, we'll analyze motion capture data from 6 different sequences
of an individual actor peforming different exercises.

Although this data is inherently sequential in nature and it is smart to 
use a model that accounts for time, we'll ignore that for now and
focus on the dataset's *grouped* nature.


That is, we can compare the following two models:

* Baseline: Gaussian mixture model that pools all observations from all sequences
* Smarter alternative: Latent Dirichlet Allocation with a Gaussian likelihood

That is, we treat each sequence as a separate collection of data examples,
modeled by *group* specific appearance probabilities but *shared* cluster
means and covariances.

"""
# sphinx_gallery_thumbnail_number = 1

import bnpy
import numpy as np
import os

from matplotlib import pylab
import seaborn as sns

FIG_SIZE = (5, 5)
LANDSCAPE_FIG_SIZE = (15, 5)
pylab.rcParams['figure.figsize'] = FIG_SIZE

np.set_printoptions(precision=3, suppress=1, linewidth=200)

###############################################################################
#
# Load dataset from file

dataset_path = os.path.join(bnpy.DATASET_PATH, 'mocap6')
dataset = bnpy.data.GroupXData.read_npz(
    os.path.join(dataset_path, 'dataset.npz'))

###############################################################################
#
# Setup: Function to make a simple plot of the raw data
# -----------------------------------------------------

def show_single_sequence(seq_id):
    start = dataset.doc_range[seq_id]
    stop = dataset.doc_range[seq_id + 1]
    pylab.figure(figsize=LANDSCAPE_FIG_SIZE)
    for dim in range(12):
        X_seq = dataset.X[start:stop]
        pylab.plot(X_seq[:, dim], '.-')
    pylab.xlabel('time')
    pylab.ylabel('angle')
    pylab.tight_layout()

###############################################################################
#
# Visualization of the first sequence
# -----------------------------------

show_single_sequence(0)


###############################################################################
#
# Setup: hyperparameters
# ----------------------------------------------------------

K = 10            # Number of clusters/states

alpha = 0.25      # group-level Dirichlet concentration parameter

gamma = 5.0       # top-level Dirichlet concentration parameter (used by HDP only)

sF = 1.0          # Set observation model prior so E[covariance] = identity
ECovMat = 'eye'

nLap = 200

###############################################################################
#
# Baseline: Mixture model with *DiagGauss* observation model
# ----------------------------------------------------------
#
# We'll take the best of 3 independent inits ('tasks')


mix_model, mix_info_dict = bnpy.run(
    dataset, 'FiniteMixtureModel', 'DiagGauss', 'memoVB',
    output_path='/tmp/mocap6/test-model=FiniteMixtureModel+DiagGauss-ECovMat=1*eye/',
    nLap=nLap, nTask=3, nBatch=1, convergeThr=0.0001,
    gamma=1.0,
    sF=sF, ECovMat=ECovMat,
    K=K, initname='randexamples',
    )


###############################################################################
#
# FiniteTopicModel with *DiagGauss* observation model
# ---------------------------------------------------
#
# We'll take the best of 3 independent inits ('tasks')


finite_model, finite_info_dict = bnpy.run(
    dataset, 'FiniteTopicModel', 'DiagGauss', 'memoVB',
    output_path='/tmp/mocap6/test-model=FiniteTopicModel+DiagGauss-ECovMat=1*eye/',
    nLap=nLap, nTask=3, nBatch=1, convergeThr=0.0001,
    alpha=alpha,
    sF=sF, ECovMat=ECovMat,
    K=K, initname='randexamples',
    )

###############################################################################
#
# HDP-HMM with *DiagGauss* observation model
# -------------------------------------------
#
# We'll take the best of 3 independent inits ('tasks')
#

hdp_topic_model, hdp_info_dict = bnpy.run(
    dataset, 'HDPTopicModel', 'DiagGauss', 'memoVB',
    output_path='/tmp/mocap6/test-model=HDPTopicModel+DiagGauss-ECovMat=1*eye/',
    nLap=nLap, nTask=3, nBatch=1, convergeThr=0.0001,
    gamma=gamma, alpha=alpha,
    sF=sF, ECovMat=ECovMat,
    K=K, initname='randexamples',
    moves='shuffle',
    )

###############################################################################
# 
# Compare loss function traces for all methods
# --------------------------------------------
#
# We'll notice that the simple mixture performs noticeably worse than the 
# more flexible models that allow group-specific cluster weights
#

pylab.figure()
pylab.plot(
    mix_info_dict['lap_history'],
    mix_info_dict['loss_history'], 'k--',
    label='mix + diag gauss')
pylab.plot(
    finite_info_dict['lap_history'],
    finite_info_dict['loss_history'], 'm.-',
    label='LDA + diag gauss')
pylab.plot(
    hdp_info_dict['lap_history'],
    hdp_info_dict['loss_history'], 'r.-',
    label='HDP + diag gauss')
pylab.legend(loc='upper right')
pylab.xlabel('num. laps')
pylab.ylabel('loss')
pylab.xlim([0, 200]) # avoid early iterations
pylab.ylim([3.5, 3.7]) # handpicked
pylab.draw()
pylab.tight_layout()

###############################################################################
# 
# Show the baseline per-sequence appearances
# -----------------------------------------------
#

np.set_printoptions(precision=3, suppress=1, linewidth=200)

## Compute approx. posterior parameter 'resp' for each example
# resp : 2D array, n_examples x n_clusters
# resp[n] defines Discrete probability of using clusters to explain each example
LP = mix_model.calc_local_params(dataset)
resp_NK = LP['resp']

## Compute the per-sequence average usage.
avg_resp_DK = np.zeros((dataset.nDoc, K))
for d in range(dataset.nDoc):
    start = dataset.doc_range[d]
    stop = dataset.doc_range[d+1]
    avg_resp_DK[d] = np.mean(resp_NK[start:stop], axis=0)

print("Baseline mixture model: per-sequence average cluster usage")
print(avg_resp_DK)

nnz = np.sum(avg_resp_DK < 0.001)
print("Sparsity level: %d/%d entries close-to-zero" % (
    nnz, avg_resp_DK.size))
print("")

###############################################################################
# 
# Show the learned group-specific mixture weights
# -----------------------------------------------
#


# Compute approx. posterior parameter 'theta' for each document
LP = finite_model.calc_local_params(dataset)
resp_NK = LP['resp']

avg_resp_DK = np.zeros((dataset.nDoc, K))
for d in range(dataset.nDoc):
    start = dataset.doc_range[d]
    stop = dataset.doc_range[d+1]
    avg_resp_DK[d] = np.mean(resp_NK[start:stop], axis=0)

print("LDA mixed membership model: per-sequence average cluster usage")
print(avg_resp_DK)
nnz = np.sum(avg_resp_DK < 0.001)
print("Sparsity level: %d/%d entries close-to-zero" % (
    nnz, avg_resp_DK.size))
print("")

# theta : 2D array, n_docs x n_clusters
# theta[d] defines Dirichlet parameters for d-th document
theta_DK = LP['theta']

# Compute expected probabilities
E_pi_DK = theta_DK / np.sum(theta_DK, axis=1)[:,np.newaxis]


print("LDA mixed membership model: per-sequence cluster probabilities")
print(E_pi_DK)

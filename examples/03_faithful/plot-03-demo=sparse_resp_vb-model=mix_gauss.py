"""
==================================================================
Demonstration of Sparse Responsibilities for Mixtures of Gaussians
==================================================================

This demo illustrates "sparse responsibilities" for mixture models.

In this example, we show how BNPy makes it easy to vary
per-example inference, so that for a 3-cluster model
we can manually try each of the following:

* each data point assigned to up to 1 cluster
* each data point assigned to up to 2 clusters
* each data point assigned to up to 3 clusters

In our variational approximation, when we fit
a K-cluster mixture model to a specific data point,
we learn a vector of posterior **responsibilities** --
we often call these "resp" for short in the code --
indicating how much of the unit probability mass for this example
is explained by each of the K clusters.

For example, for a specific data point we might have a resp vector of:

    [ 0.      0.9262  0.0738]

which we interpret as saying this point is 92% explained by the 
2nd cluster, 7% by the 3rd cluster, and 0% by the first.

This short demo just shows how we can deliberately enforce sparsity
in learned responsibilities, so that each example may have only L 
non-zero entries in its resp vector. For example, we'll fit a K=3 model
below, and show how we can enforce L=1 or L=2 sparsity. 

For a technical introduction, see our ArXiv paper:

    Michael C. Hughes and Erik B. Sudderth (2016)
    "Fast Learning of Clusters and Topics via Sparse Posteriors."
    https://arxiv.org/abs/1609.07521


"""
from __future__ import print_function

import bnpy
import numpy as np
import os

from matplotlib import pylab
import seaborn as sns

SMALL_FIG_SIZE = (2.5, 2.5)
FIG_SIZE = (5, 5)
pylab.rcParams['figure.figsize'] = FIG_SIZE

###############################################################################
#
# Load dataset from file

dataset_path = os.path.join(bnpy.DATASET_PATH, 'faithful')
dataset = bnpy.data.XData.read_csv(
    os.path.join(dataset_path, 'faithful.csv'))

# Identify "target" example ids to focus on
target_example_ids = np.asarray(
    [ 83,  46, 152, 173,  23, 164, 243, 239, 154, 100])
targeted_dataset = bnpy.data.XData(
    dataset.X[target_example_ids])

###############################################################################
#
# Make a simple plot of the raw data

pylab.plot(dataset.X[:, 0], dataset.X[:, 1], 'k.')
pylab.xlabel(dataset.column_names[0])
pylab.ylabel(dataset.column_names[1])
pylab.tight_layout()
pylab.title('Complete Dataset')
data_ax_h = pylab.gca()


###############################################################################
#
# Setup: Func to print resp array nicely
# --------------------------------------

def pprint_sparse_or_dense_resp(LP, target_example_ids=target_example_ids):
    np.set_printoptions(suppress=True, precision=4)
    if 'spR' in LP:
        # sparse case
        resp_NK = LP['spR'][target_example_ids].toarray()
    else:
        resp_NK = LP['resp'][target_example_ids]
    print(resp_NK)


###############################################################################
#
# VB with DP mixture model and diagonal Gaussian observations
# -----------------------------------------------------------
#
# Assumes exactly 3 clusters
# 
# Assumes diagonal covariances.
# 
# No sparsity assumptions during training

K = 3          # n clusters
gamma = 50.0   # DP concentration param
sF = 0.1       # scale of expected covariance

full_trained_model, full_info_dict = bnpy.run(
    dataset, 'DPMixtureModel', 'DiagGauss', 'VB',
    output_path='/tmp/faithful/demo_sparse_resp-K=3-lik=Gauss-ECovMat=5*eye/',
    nLap=1000, nTask=5, nBatch=1, convergeThr=0.0001,
    gamma0=gamma, sF=sF, ECovMat='eye',
    K=K, initname='randexamples',
    )

# Add this model into the current plot
bnpy.viz.PlotComps.plotCompsFromHModel(
    full_trained_model,
    )

###############################################################################
#
# Do inference with L=1 sparsity
# ------------------------------
#
# Kwarg to set is "nnzPerRowLP = 1"
#
# nnzPerRowLP is read as "number of non-zero entries per row of local parameters"
#
# This enforces that each row of resp array has ___ non zero entries

print('SPARSITY LEVEL: 1 of 3')
LP1 = full_trained_model.calc_local_params(dataset, nnzPerRowLP=1)
pprint_sparse_or_dense_resp(LP1)


###############################################################################
#
# Do inference with L=2 sparsity
# ------------------------------

print('SPARSITY LEVEL: 2 of 3')
LP2 = full_trained_model.calc_local_params(dataset, nnzPerRowLP=2)
pprint_sparse_or_dense_resp(LP2)


###############################################################################
#
# Do inference with L=3 sparsity
# ------------------------------


print('SPARSITY LEVEL: 3 of 3 (DENSE)')
LP3 = full_trained_model.calc_local_params(dataset, nnzPerRowLP=3)
pprint_sparse_or_dense_resp(LP3)


# Show the model
pylab.figure(2)
bnpy.viz.PlotComps.plotCompsFromHModel(
    full_trained_model,
    dataset=targeted_dataset,
    )
pylab.xlabel(dataset.column_names[0])
pylab.ylabel(dataset.column_names[1])
pylab.tight_layout()
pylab.title('Trained model with targeted examples')


###############################################################################
#
# How to incorporate sparsity during training?
# --------------------------------------------
# Just pass "nnzPerRowLP" kwarg to bnpy.run(...)

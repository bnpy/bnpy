"""
====================================
Comparing models for sequential data
====================================

How to train mixtures and HMMs with various observation models on the same dataset.

"""
# sphinx_gallery_thumbnail_number = 1

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
# Visualization of the second sequence
# ------------------------------------

show_single_sequence(1)

###############################################################################
#
# Setup: hyperparameters
# ----------------------------------------------------------

K = 20            # Number of clusters/states

gamma = 5.0       # top-level Dirichlet concentration parameter
transAlpha = 0.5  # trans-level Dirichlet concentration parameter 

sF = 1.0          # Set observation model prior so E[covariance] = identity
ECovMat = 'eye'

###############################################################################
#
# DP mixture with *DiagGauss* observation model
# ---------------------------------------------


mixdiag_trained_model, mixdiag_info_dict = bnpy.run(
    dataset, 'DPMixtureModel', 'DiagGauss', 'memoVB',
    output_path='/tmp/mocap6/showcase-K=20-model=DP+DiagGauss-ECovMat=1*eye/',
    nLap=50, nTask=1, nBatch=1, convergeThr=0.0001,
    gamma=gamma, sF=sF, ECovMat=ECovMat,
    K=K, initname='randexamples',
    )

###############################################################################
#
# HDP-HMM with *DiagGauss* observation model
# -------------------------------------------
#
# Assume diagonal covariances.
#
# Start with too many clusters (K=20)


hmmdiag_trained_model, hmmdiag_info_dict = bnpy.run(
    dataset, 'HDPHMM', 'DiagGauss', 'memoVB',
    output_path='/tmp/mocap6/showcase-K=20-model=HDPHMM+DiagGauss-ECovMat=1*eye/',
    nLap=50, nTask=1, nBatch=1, convergeThr=0.0001,
    transAlpha=transAlpha, gamma=gamma, sF=sF, ECovMat=ECovMat,
    K=K, initname='randexamples',
    )

###############################################################################
#
# HDP-HMM with *Gauss* observation model
# --------------------------------------
#
# Assume full covariances.
#
# Start with too many clusters (K=20)


hmmfull_trained_model, hmmfull_info_dict = bnpy.run(
    dataset, 'HDPHMM', 'Gauss', 'memoVB',
    output_path='/tmp/mocap6/showcase-K=20-model=HDPHMM+Gauss-ECovMat=1*eye/',
    nLap=50, nTask=1, nBatch=1, convergeThr=0.0001,
    transAlpha=transAlpha, gamma=gamma, sF=sF, ECovMat=ECovMat,
    K=K, initname='randexamples',
    )

###############################################################################
#
# HDP-HMM with *AutoRegGauss* observation model
# ----------------------------------------------
#
# Assume full covariances.
#
# Start with too many clusters (K=20)


hmmar_trained_model, hmmar_info_dict = bnpy.run(
    dataset, 'HDPHMM', 'AutoRegGauss', 'memoVB',
    output_path='/tmp/mocap6/showcase-K=20-model=HDPHMM+AutoRegGauss-ECovMat=1*eye/',
    nLap=50, nTask=1, nBatch=1, convergeThr=0.0001,
    transAlpha=transAlpha, gamma=gamma, sF=sF, ECovMat=ECovMat,
    K=K, initname='randexamples',
    )


###############################################################################
# 
# Compare loss function traces for all methods
# --------------------------------------------
#
pylab.figure()

pylab.plot(
    mixdiag_info_dict['lap_history'],
    mixdiag_info_dict['loss_history'], 'b.-',
    label='mix + diag gauss')
pylab.plot(
    hmmdiag_info_dict['lap_history'],
    hmmdiag_info_dict['loss_history'], 'k.-',
    label='hmm + diag gauss')
pylab.plot(
    hmmfull_info_dict['lap_history'],
    hmmfull_info_dict['loss_history'], 'r.-',
    label='hmm + full gauss')
pylab.plot(
    hmmar_info_dict['lap_history'],
    hmmar_info_dict['loss_history'], 'c.-',
    label='hmm + ar gauss')
pylab.legend(loc='upper right')
pylab.xlabel('num. laps')
pylab.ylabel('loss')
pylab.xlim([4, 100]) # avoid early iterations
pylab.ylim([2.4, 3.7]) # handpicked
pylab.draw()
pylab.tight_layout()

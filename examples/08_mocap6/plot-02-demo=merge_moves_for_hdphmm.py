"""
========================
Merge moves with HDP-HMM
========================

How to try merge moves efficiently for time-series datasets.

This example reviews three possible ways to plan and execute merge
proposals.

* try merging all pairs of clusters
* pick fewer merge pairs (at most 5 per cluster) in a size-biased way
* pick fewer merge pairs (at most 5 per cluster) in objective-driven way

"""
# sphinx_gallery_thumbnail_number = 2

import bnpy
import numpy as np
import os

from matplotlib import pylab
import seaborn as sns

FIG_SIZE = (10, 5)
pylab.rcParams['figure.figsize'] = FIG_SIZE

###############################################################################
#
# Setup: Load data
# ----------------

# Read bnpy's built-in "Mocap6" dataset from file.

dataset_path = os.path.join(bnpy.DATASET_PATH, 'mocap6')
dataset = bnpy.data.GroupXData.read_npz(
    os.path.join(dataset_path, 'dataset.npz'))



###############################################################################
#
# Setup: Initialization hyperparameters
# -------------------------------------

init_kwargs = dict(
    K=20,
    initname='randexamples',
    )

alg_kwargs = dict(
    nLap=29,
    nTask=1, nBatch=1, convergeThr=0.0001,
    )

###############################################################################
#
# Setup: HDP-HMM hyperparameters
# ------------------------------

hdphmm_kwargs = dict(
    gamma = 5.0,       # top-level Dirichlet concentration parameter
    transAlpha = 0.5,  # trans-level Dirichlet concentration parameter 
    )

###############################################################################
#
# Setup: Gaussian observation model hyperparameters
# -------------------------------------------------

gauss_kwargs = dict(
    sF = 1.0,          # Set prior so E[covariance] = identity
    ECovMat = 'eye',
    )


###############################################################################
#
# All-Pairs : Try all possible pairs of merges every 10 laps
# ----------------------------------------------------------
#
# This is expensive, but a good exhaustive test.

allpairs_merge_kwargs = dict(
    m_startLap = 10,
    # Set limits to number of merges attempted each lap.
    # This value specifies max number of tries for each cluster
    # Setting this very high (to 50) effectively means try all pairs
    m_maxNumPairsContainingComp = 50,
    # Set "reactivation" limits
    # So that each cluster is eligible again after 10 passes thru dataset
    # Or when it's size changes by 400%
    m_nLapToReactivate = 10,
    m_minPercChangeInNumAtomsToReactivate = 400 * 0.01,
    # Specify how to rank pairs (determines order in which merges are tried)
    # 'total_size' and 'descending' means try largest combined clusters first
    m_pair_ranking_procedure = 'total_size',
    m_pair_ranking_direction = 'descending',
    )

allpairs_trained_model, allpairs_info_dict = bnpy.run(
    dataset, 'HDPHMM', 'DiagGauss', 'memoVB',
    output_path='/tmp/mocap6/trymerge-K=20-model=HDPHMM+DiagGauss-ECovMat=1*eye-merge_strategy=all_pairs/',
    moves='merge,shuffle',
    **dict(
        sum(map(list,   [alg_kwargs.items(),
                        init_kwargs.items(),
                        hdphmm_kwargs.items(),
                        gauss_kwargs.items(),
                        allpairs_merge_kwargs.items()]),[]))
)

###############################################################################
#
# Large-Pairs : Try 5-largest-size pairs of merges every 10 laps
# --------------------------------------------------------------
#
# This is much cheaper than all pairs. Let's see how well it does.

largepairs_merge_kwargs = dict(
    m_startLap = 10,
    # Set limits to number of merges attempted each lap.
    # This value specifies max number of tries for each cluster
    m_maxNumPairsContainingComp = 5,
    # Set "reactivation" limits
    # So that each cluster is eligible again after 10 passes thru dataset
    # Or when it's size changes by 400%
    m_nLapToReactivate = 10,
    m_minPercChangeInNumAtomsToReactivate = 400 * 0.01,
    # Specify how to rank pairs (determines order in which merges are tried)
    # 'total_size' and 'descending' means try largest size clusters first
    m_pair_ranking_procedure = 'total_size',
    m_pair_ranking_direction = 'descending',
    )


largepairs_trained_model, largepairs_info_dict = bnpy.run(
    dataset, 'HDPHMM', 'DiagGauss', 'memoVB',
    output_path='/tmp/mocap6/trymerge-K=20-model=HDPHMM+DiagGauss-ECovMat=1*eye-merge_strategy=large_pairs/',
    moves='merge,shuffle',
    **dict(
        sum(map(list,   [alg_kwargs.items(),
                        init_kwargs.items(),
                        hdphmm_kwargs.items(),
                        gauss_kwargs.items(),
                        largepairs_merge_kwargs.items()]),[])))

###############################################################################
#
# Good-ELBO-Pairs : Rank pairs of merges by improvement to observation model
# --------------------------------------------------------------------------
#
# This is much cheaper than all pairs and perhaps more principled.
# Let's see how well it does.

goodelbopairs_merge_kwargs = dict(
    m_startLap = 10,
    # Set limits to number of merges attempted each lap.
    # This value specifies max number of tries for each cluster
    m_maxNumPairsContainingComp = 5,
    # Set "reactivation" limits
    # So that each cluster is eligible again after 10 passes thru dataset
    # Or when it's size changes by 400%
    m_nLapToReactivate = 10,
    m_minPercChangeInNumAtomsToReactivate = 400 * 0.01,
    # Specify how to rank pairs (determines order in which merges are tried)
    # 'obsmodel_elbo' means rank pairs by improvement to observation model ELBO
    m_pair_ranking_procedure = 'obsmodel_elbo',
    m_pair_ranking_direction = 'descending',
    )


goodelbopairs_trained_model, goodelbopairs_info_dict = bnpy.run(
    dataset, 'HDPHMM', 'DiagGauss', 'memoVB',
    output_path='/tmp/mocap6/trymerge-K=20-model=HDPHMM+DiagGauss-ECovMat=1*eye-merge_strategy=good_elbo_pairs/',
    moves='merge,shuffle',
    **dict(
        sum(map(list,   [alg_kwargs.items(),
                        init_kwargs.items(),
                        hdphmm_kwargs.items(),
                        gauss_kwargs.items(),
                        goodelbopairs_merge_kwargs.items()]),[])))



###############################################################################
# 
# Compare loss function vs wallclock time
# ---------------------------------------
#
pylab.figure()
for info_dict, color_str, label_str in [
        (allpairs_info_dict, 'k', 'all_pairs'),
        (largepairs_info_dict, 'g', 'large_pairs'),
        (goodelbopairs_info_dict, 'b', 'good_elbo_pairs')]:
    pylab.plot(
        info_dict['elapsed_time_sec_history'],
        info_dict['loss_history'],
        '.-',
        color=color_str,
        label=label_str)
pylab.legend(loc='upper right')
pylab.xlabel('elapsed time (sec)')
pylab.ylabel('loss')


###############################################################################
# 
# Compare number of active clusters vs wallclock time
# ---------------------------------------------------
#
pylab.figure()
for info_dict, color_str, label_str in [
        (allpairs_info_dict, 'k', 'all_pairs'),
        (largepairs_info_dict, 'g', 'large_pairs'),
        (goodelbopairs_info_dict, 'b', 'good_elbo_pairs')]:
    pylab.plot(
        info_dict['elapsed_time_sec_history'],
        info_dict['K_history'],
        '.-',
        color=color_str,
        label=label_str)
pylab.legend(loc='upper right')
pylab.xlabel('elapsed time (sec)')
pylab.ylabel('num. clusters (K)')

pylab.show(block=False)

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

nTask = 2

###############################################################################
#
# Let's run the VB coordinate ascent with proposal moves.
# 
# Using 1 initial cluster.

trained_model, info_dict = bnpy.run(
    dataset, 'DPMixtureModel', 'Gauss', 'memoVB',
    output_path='/tmp/AsteriskK8/trymoves-K=1/',
    nLap=100, nTask=nTask, nBatch=1,
    sF=0.1, ECovMat='eye',
    K=1, initname='randexamples',
    moves='birth,merge,shuffle',
    m_startLap=5, b_startLap=2, b_Kfresh=4)

###############################################################################
#
# Now using 4 initial clusters

trained_model, info_dict = bnpy.run(
    dataset, 'DPMixtureModel', 'Gauss', 'memoVB',
    output_path='/tmp/AsteriskK8/trymoves-K=4/',
    nLap=100, nTask=nTask, nBatch=1,
    sF=0.1, ECovMat='eye',
    K=4, initname='randexamples',
    moves='birth,merge,shuffle',
    m_startLap=5, b_startLap=2, b_Kfresh=4)

###############################################################################
# 
# Now using 8 initial clusters

trained_model, info_dict = bnpy.run(
    dataset, 'DPMixtureModel', 'Gauss', 'memoVB',
    output_path='/tmp/AsteriskK8/trymoves-K=8/',
    nLap=100, nTask=nTask, nBatch=1,
    sF=0.1, ECovMat='eye',
    K=8, initname='randexamples',
    moves='birth,merge,shuffle',
    m_startLap=5, b_startLap=2, b_Kfresh=4)

###############################################################################
# 
# Now using 25 initial clusters

trained_model, info_dict = bnpy.run(
    dataset, 'DPMixtureModel', 'Gauss', 'memoVB',
    output_path='/tmp/AsteriskK8/trymoves-K=25/',
    nLap=100, nTask=nTask, nBatch=1,
    sF=0.1, ECovMat='eye',
    K=25, initname='randexamples',
    moves='birth,merge,shuffle',
    m_startLap=5, b_startLap=2, b_Kfresh=4)

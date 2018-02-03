"""
=================================================
VB coordinate descent for Mixture of Multinomials
=================================================


"""
import bnpy
import numpy as np
import os

from matplotlib import pylab
import seaborn as sns

FIG_SIZE = (3, 3)
SMALL_FIG_SIZE = (1,1)
pylab.rcParams['figure.figsize'] = FIG_SIZE

top_word_kws = dict(
    wordSizeLimit=15,
    ncols=4,
    Ktop=10)

###############################################################################
# Read text dataset from file

dataset_path = os.path.join(bnpy.DATASET_PATH, 'we8there', 'raw')
dataset = bnpy.data.BagOfWordsData.read_npz(
    os.path.join(dataset_path, 'dataset.npz'),
    vocabfile=os.path.join(dataset_path, 'x_csc_colnames.txt'))

# Filter out documents with less than 20 words
doc_ids = np.flatnonzero(
    dataset.getDocTypeCountMatrix().sum(axis=1) >= 20)
dataset = dataset.make_subset(docMask=doc_ids, doTrackFullSize=False)

###############################################################################
#
# Make a simple plot of the raw data
bnpy.viz.PrintTopics.plotCompsFromWordCounts(
    dataset.getDocTypeCountMatrix()[:10],
    vocabList=dataset.vocabList,
    prefix='doc',
    **top_word_kws)

###############################################################################
#
# Train with K=1 cluster
# ----------------------
# 
# This is a simple baseline.

trained_model, info_dict = bnpy.run(
    dataset, 'DPMixtureModel', 'Mult', 'VB',
    output_path='/tmp/we8there/helloworld-model=dp_mix+mult-K=1/',
    nLap=1000, convergeThr=0.0001, nTask=1,
    K=1, initname='bregmankmeans+lam1+iter1',
    gamma0=50.0, lam=0.1)

bnpy.viz.PrintTopics.plotCompsFromHModel(
    trained_model,
    vocabList=dataset.vocabList,
    **top_word_kws)

###############################################################################
#
# Train with K=3 clusters
# -----------------------
#
# Take the best of 10 initializations

trained_model, info_dict = bnpy.run(
    dataset, 'DPMixtureModel', 'Mult', 'VB',
    output_path='/tmp/we8there/helloworld-model=dp_mix+mult-K=3/',
    nLap=1000, convergeThr=0.0001, nTask=10,
    K=3, initname='bregmankmeans+lam1+iter1',
    gamma0=50.0, lam=0.1)

bnpy.viz.PrintTopics.plotCompsFromHModel(
    trained_model,
    vocabList=dataset.vocabList,
    **top_word_kws)


###############################################################################
#
# Train with K=10 clusters
# ------------------------
# 
# Take the best of 10 initializations

trained_model, info_dict = bnpy.run(
    dataset, 'DPMixtureModel', 'Mult', 'VB',
    output_path='/tmp/we8there/helloworld-model=dp_mix+mult-K=10/',
    nLap=1000, convergeThr=0.0001, nTask=10,
    K=10, initname='bregmankmeans+lam1+iter1',
    gamma0=50.0, lam=0.1)

bnpy.viz.PrintTopics.plotCompsFromHModel(
    trained_model,
    vocabList=dataset.vocabList,
    **top_word_kws)

###############################################################################
#
# Train with K=30 clusters
# ------------------------
# 
# Take the best of 10 initializations

trained_model, info_dict = bnpy.run(
    dataset, 'DPMixtureModel', 'Mult', 'VB',
    output_path='/tmp/we8there/helloworld-model=dp_mix+mult-K=30/',
    nLap=1000, convergeThr=0.0001, nTask=10,
    K=30, initname='bregmankmeans+lam1+iter1',
    gamma0=50.0, lam=0.1)

bnpy.viz.PrintTopics.plotCompsFromHModel(
    trained_model,
    vocabList=dataset.vocabList,
    **top_word_kws)

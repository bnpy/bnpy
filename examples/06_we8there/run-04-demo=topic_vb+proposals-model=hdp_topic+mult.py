"""
====================================================
Birth and merge variational training for topic model
====================================================


"""
import bnpy
import numpy as np
import os

from matplotlib import pylab
import seaborn as sns

FIG_SIZE = (2, 2)
SMALL_FIG_SIZE = (1.5, 1.5)

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
# Train LDA topic model
# ---------------------
# 
# Using 10 clusters and a random initialization procedure.

local_step_kwargs = dict(
    # perform at most this many iterations at each document
    nCoordAscentItersLP=100,
    # stop local iters early when max change in doc-topic counts < this thr
    convThrLP=0.001,
    )
merge_kwargs = dict(
    m_startLap=5,
    )
birth_kwargs = dict(
    b_startLap=2,
    b_stopLap=20,
    b_Kfresh=5)

trained_model, info_dict = bnpy.run(
    dataset, 'HDPTopicModel', 'Mult', 'memoVB',
    output_path='/tmp/we8there/trymoves-model=hdp_topic+mult-K=5/',
    nLap=20, convergeThr=0.01, nBatch=1,
    K=5, initname='randomlikewang',
    gamma=50.0, alpha=0.5, lam=0.1,
    moves='birth,merge,shuffle',
    **dict(list(local_step_kwargs.items()) + 
        list(merge_kwargs.items()) + 
        list(birth_kwargs.items())))

###############################################################################
#
# Setup: Helper function to plot topics at each stage of training

def show_top_words_over_time(
        task_output_path=None,
        vocabList=None,
        query_laps=[0, 1, 2, 5, None],
        ncols=10):
    '''
    '''
    nrows = len(query_laps)
    fig_handle, ax_handles_RC = pylab.subplots(
        figsize=(SMALL_FIG_SIZE[0] * ncols, SMALL_FIG_SIZE[1] * nrows),
        nrows=nrows, ncols=ncols, sharex=True, sharey=True)
    for row_id, lap_val in enumerate(query_laps):
        cur_model, lap_val = bnpy.load_model_at_lap(task_output_path, lap_val)
        # Plot the current model
        cur_ax_list = ax_handles_RC[row_id].flatten().tolist()
        bnpy.viz.PrintTopics.plotCompsFromHModel(
            cur_model,
            vocabList=vocabList,
            fontsize=9,
            Ktop=7,
            ax_list=cur_ax_list)
        cur_ax_list[0].set_ylabel("lap: %d" % lap_val)
    pylab.subplots_adjust(
        wspace=0.04, hspace=0.1, 
        left=0.01, right=0.99, top=0.99, bottom=0.1)
    pylab.tight_layout()


###############################################################################
#
# Show the topics over time

show_top_words_over_time(
    info_dict['task_output_path'], vocabList=dataset.vocabList)

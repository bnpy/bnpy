"""
=================================================
03: Standard variational training for topic model
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

###############################################################################
# Read dataset from file.

dataset_path = os.path.join(bnpy.DATASET_PATH, 'bars_one_per_doc')
dataset = bnpy.data.BagOfWordsData.read_npz(
    os.path.join(dataset_path, 'dataset.npz'))

###############################################################################
#
# Make a simple plot of the raw data
X_csr_DV = dataset.getSparseDocTypeCountMatrix()
bnpy.viz.BarsViz.show_square_images(
    X_csr_DV[:10].toarray(), vmin=0, vmax=5)
pylab.tight_layout()

###############################################################################
#
# Setup: Function to show bars from start to end of training run

def show_bars_over_time(
        task_output_path=None,
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
        cur_topics_KV = cur_model.obsModel.getTopics()
        # Plot the current model
        cur_ax_list = ax_handles_RC[row_id].flatten().tolist()
        bnpy.viz.BarsViz.show_square_images(
            cur_topics_KV,
            vmin=0.0, vmax=0.06,
            ax_list=cur_ax_list)
        cur_ax_list[0].set_ylabel("lap: %d" % lap_val)
    pylab.tight_layout()

###############################################################################
# Train LDA topic model
# ---------------------
# 
# Using 10 clusters and the 'randexamples' initialization procedure.

local_step_kwargs = dict(
    # perform at most this many iterations at each document
    nCoordAscentItersLP=100,
    # stop local iters early when max change in doc-topic counts < this thr
    convThrLP=0.001,
    )

trained_model, info_dict = bnpy.run(
    dataset, 'FiniteTopicModel', 'Mult', 'VB',
    output_path='/tmp/bars_one_per_doc/helloworld-model=topic+mult-K=10/',
    nLap=100, convergeThr=0.01,
    K=10, initname='randomlikewang',
    alpha=0.5, lam=0.1,
    **local_step_kwargs)

###############################################################################
#
# First, we can plot the loss function over time
# We'll skip the first few iterations, since performance is quite bad.
#

pylab.figure(figsize=FIG_SIZE)
pylab.plot(info_dict['lap_history'][1:], info_dict['loss_history'][1:], 'k.-')
pylab.xlabel('num. laps')
pylab.ylabel('loss')
pylab.tight_layout()

###############################################################################
#
# Show the clusters over time
show_bars_over_time(info_dict['task_output_path'])


###############################################################################
# Train LDA topic model with restarts
# -----------------------------------
# 
# Using 10 clusters and the 'randexamples' initialization procedure.

r_local_step_kwargs = dict(
    # perform at most this many iterations at each document
    nCoordAscentItersLP=100,
    # stop local iters early when max change in doc-topic counts < this thr
    convThrLP=0.001,
    # perform restart proposals at each document
    restartLP=1,
    restartNumItersLP=5,
    restartNumTrialsLP=5,
    )

r_trained_model, r_info_dict = bnpy.run(
    dataset, 'FiniteTopicModel', 'Mult', 'VB',
    output_path='/tmp/bars_one_per_doc/helloworld-model=topic+mult-K=10-localstep=restarts/',
    nLap=100, convergeThr=0.01,
    K=10, initname='randomlikewang',
    alpha=0.5, lam=0.1,
    **r_local_step_kwargs)

show_bars_over_time(r_info_dict['task_output_path'])

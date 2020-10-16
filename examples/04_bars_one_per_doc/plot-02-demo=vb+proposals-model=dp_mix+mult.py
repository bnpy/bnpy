"""
============================================================
02: Training DP mixture model with birth and merge proposals
============================================================

How to train a DP mixture of multinomials.
"""
import bnpy
import numpy as np
import os

from matplotlib import pylab
import seaborn as sns

FIG_SIZE = (3, 3)
SMALL_FIG_SIZE = (1.5, 1.5)
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
            vmin=0.0, vmax=0.1,
            ax_list=cur_ax_list)
        cur_ax_list[0].set_ylabel("lap: %d" % lap_val)
    pylab.tight_layout()

###############################################################################
# From K=2 initial clusters
# -------------------------
# 
# Using random initialization

initname = 'randomlikewang'
K = 2

K2_trained_model, K2_info_dict = bnpy.run(
    dataset, 'DPMixtureModel', 'Mult', 'memoVB',
    output_path='/tmp/bars_one_per_doc/trymoves-K=%d-initname=%s/' % (
        K, initname),
    nTask=1, nLap=50, convergeThr=0.0001, nBatch=1,
    K=K, initname=initname,
    gamma0=50.0, lam=0.1,
    moves='birth,merge,shuffle,delete',
    b_startLap=2,
    m_startLap=5,
    d_startLap=10)

show_bars_over_time(K2_info_dict['task_output_path'])



###############################################################################
# K=10 initial clusters
# ---------------------
# 
# Using random initialization

initname = 'randomlikewang'
K = 10

K10_trained_model, K10_info_dict = bnpy.run(
    dataset, 'DPMixtureModel', 'Mult', 'memoVB',
    output_path='/tmp/bars_one_per_doc/trymoves-K=%d-initname=%s/' % (
        K, initname),
    nTask=1, nLap=50, convergeThr=0.0001, nBatch=1,
    K=K, initname=initname,
    gamma0=50.0, lam=0.1,
    moves='birth,merge,shuffle,delete',
    b_startLap=2,
    m_startLap=5,
    d_startLap=10)

show_bars_over_time(K10_info_dict['task_output_path'])



###############################################################################
# Draw sample documents from learned model
# ----------------------------------------
# 

# Create random number generator with fixed seed
prng = np.random.RandomState(54321)

# Preallocate space for 5 generated docs
n_docs_to_sample = 5
V = X_csr_DV.shape[1]
test_x_DV = np.zeros((n_docs_to_sample, V))

for doc_id in range(n_docs_to_sample):
    # Step 1: Pick cluster index *k* that current example is assigned to
    proba_K = K10_trained_model.allocModel.get_active_comp_probs()
    k = prng.choice(proba_K.size, p=proba_K / np.sum(proba_K))

    # Step 2: Draw probability-over-vocab from cluster *k*'s Dirichlet posterior
    lam_k_V = K10_trained_model.obsModel.Post.lam[k]
    phi_k_V = prng.dirichlet(lam_k_V)

    # Step 3: Draw a document with 50 words using phi_k_V
    x_d_V = prng.multinomial(50, phi_k_V)
    test_x_DV[doc_id] = x_d_V

bnpy.viz.BarsViz.show_square_images(
    test_x_DV, vmin=0, vmax=5)
pylab.tight_layout()
pylab.show(block=False)

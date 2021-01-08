"""
====================================================
Scalable training of HDP topic models
====================================================

In this demo, we'll review the scalable memoized training of HDP topic models.

To review, our memoized VB algorithm (Hughes and Sudderth, NeurIPS 2013) proceeds like this pseudocode:

.. code-block:: python

    n_laps_completed = 0
    while n_laps_completed < nLap:
    
        n_batches_completed_this_lap = 0
        while n_batches_completed_this_lap < nBatch:
        
            batch_data = next_minibatch()

            # Batch-specific local step
            LPbatch = model.calc_local_params(batch_data, **local_step_kwargs)

            # Batch-specific summary step
            SSbatch = model.get_summary_stats(batch_data, LPbatch)

            # Increment global summary statistics
            SS = update_global_stats(SS, SSbatch)

            # Update global parameters
            model.update_global_params(SS)


From a runtime perspective, the important settings a user can control are:

* nBatch: the number of batches
* nLap  : the number of required laps (passes thru full dataset) to perform
* local_step_kwargs : dict of keyword arguments that control local step optimization

What happens at each step?
--------------------------
In the local step, we visit each document in the current batch.
At each document, we estimate its local (document-specific) variational posterior.
This is done via an *iterative* algorithm, which is rather expensive.
We might need 50 or 100 or 200 iterations at each document, though each iteration is linear in the number of documents and the number of topics.

The summary step simply computes the sufficient statistics for the batch.
Usually this is far faster than the local step, since it a closed-form computation not an iterative estimation.

The global parameter update step is similarly quite fast, because we're using a model that enjoys conjugacy (e.g. the observation model's global posterior is a Dirichlet, related to a Multinomial likelihood and a Dirichlet prior). 

Thus, the *local step* is the runtime bottleneck.


Runtime vs nBatch
-----------------
It may be tempting to think that smaller minibatches (increasing nBatch) will make the code go "faster".
However, if you fix the number of laps to be completed, increasing the number of batches leads to strictly *more* work.

However, for each of the requested laps, here's the work performed:

* the *same* number of per-document local update iterations are completed
* the *same* number of per-document summaries are completed
* the total number of global parameter updates is exactly nBatch

For scaling to large datasets, the important thing is *not* to keep the number of laps the same, but to keep the wallclock runtime the same, and then to ask how much progress is made in reducing the loss (either training loss or validation loss, whichever is more relevant). Running with larger nBatch values will usually give improved progress in the same amount of time.


Runtime vs Local Step Convergence Thresholds
--------------------------------------------
Since the local step dominates the cost of updates, managing the run time of the local iterations is important.

There are two settings in the code that control this:

* nCoordAscentItersLP : number of local step iterations to perform per document
* convThrLP : threshold to decide if local step updates have converged

The local step pseudocode is:

.. code-block:: python

    for each document d:
    
        for iter in [1, 2, ..., nCoordAscentItersLP]:
    
            # Update q(\pi_d), the variational posterior for document d's
            # topic probability vector
            
            # Update q(z_d), the variational posterior for document d's
            # topic-word discrete assignments

            # Compute N_d1, ... N_dK, expected count of topic k in document d

            if iter % 5 == 0: # every 5 iterations, check for early convergence

                # Quit early if no N_dk entry changes by more than convThrLP
```

Thus, setting these local step optimization hyperparameters can be very practically important.

Setting convThrLP to -1 (or any number less than zero) will always do all the requested iterations.
Setting convThrLP to something moderate (like 0.05) will often reduce the local step cost by 2x or more.

"""

import bnpy
import numpy as np
import os

import matplotlib.pyplot as plt

###############################################################################
#
# Read text dataset from file
#
# Keep the first 6400 documents so we have a nice even number

dataset_path = os.path.join(bnpy.DATASET_PATH, 'wiki')
dataset = bnpy.data.BagOfWordsData.LoadFromFile_ldac(
    os.path.join(dataset_path, 'train.ldac'),
    vocabfile=os.path.join(dataset_path, 'vocab.txt'))
 
# Keep 6400 documents with at least 50 words
doc_ids = np.flatnonzero(dataset.getDocTypeCountMatrix().sum(axis=1) >= 50)
dataset = dataset.make_subset(docMask=doc_ids[:6400], doTrackFullSize=False)

###############################################################################
# Train scalable HDP topic models
# -------------------------------
# 
# Vary the number of batches and the local step convergence threshold

# Model kwargs
gamma = 25.0
alpha = 0.5
lam = 0.1

# Initialization kwargs
K = 25 

# Algorithm kwargs
nLap = 5
traceEvery = 0.5
printEvery = 0.5
convThr = 0.01

for row_id, convThrLP in enumerate([-1.00, 0.25]):

    local_step_kwargs = dict(
        # perform at most this many iterations at each document
        nCoordAscentItersLP=100,
        # stop local iters early when max change in doc-topic counts < this thr
        convThrLP=convThrLP,
        )

    for nBatch in [1, 16]:
        
        output_path = '/tmp/wiki/scalability-model=hdp_topic+mult-alg=memoized-nBatch=%d-nCoordAscentItersLP=%s-convThrLP=%.3g/' % (
                nBatch, local_step_kwargs['nCoordAscentItersLP'], convThrLP)

        trained_model, info_dict = bnpy.run(
            dataset, 'HDPTopicModel', 'Mult', 'memoVB',
            output_path=output_path,
            nLap=nLap, nBatch=nBatch, convThr=convThr,
            K=K, gamma=gamma, alpha=alpha, lam=lam,
            initname='randomlikewang', 
            moves='shuffle',
            traceEvery=traceEvery, printEvery=printEvery,
            **local_step_kwargs)


###############################################################################
# Plot: Training Loss and Laps Completed vs. Wallclock time
# ---------------------------------------------------------
#
# * Left column: Training Loss progress vs. wallclock time
# * Right column: Laps completed vs. wallclock time
#
# Remember: one lap is a complete pass through entire training set (6400 docs)

H = 3; W = 4
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(2*W,2*H), sharex=True, sharey=False)    
 
for row_id, convThrLP in enumerate([-1.00, 0.25]):
        
    for nBatch in [1, 16]:

        output_path = '/tmp/wiki/scalability-model=hdp_topic+mult-alg=memoized-nBatch=%d-nCoordAscentItersLP=%s-convThrLP=%.3g/' % (
            nBatch, local_step_kwargs['nCoordAscentItersLP'], convThrLP)

        elapsed_time_T = np.loadtxt(os.path.join(output_path, '1', 'trace_elapsed_time_sec.txt'))
        elapsed_laps_T = np.loadtxt(os.path.join(output_path, '1', 'trace_lap.txt'))
        loss_T = np.loadtxt(os.path.join(output_path, '1', 'trace_loss.txt'))
    
        ax[row_id, 0].plot(elapsed_time_T, loss_T, '.-', label='nBatch=%d, batch_size = %d' % (nBatch, 6400/nBatch))
        ax[row_id, 1].plot(elapsed_time_T, elapsed_laps_T, '.-', label='nBatch=%d' % nBatch)

        ax[row_id, 0].set_ylabel('training loss')
        ax[row_id, 1].set_ylabel('laps completed')

        ax[row_id, 0].set_xlabel('elapsed time (sec)')
        ax[row_id, 1].set_xlabel('elapsed time (sec)')
    ax[row_id, 0].legend(loc='upper right')
    ax[row_id, 0].set_title(('Loss vs Time, local conv. thr. %.2f' % (convThrLP)).replace(".00", ""))
    ax[row_id, 1].set_title(('Laps vs Time, local conv. thr. %.2f' % (convThrLP)).replace(".00", ""))

plt.tight_layout()
plt.show()

###############################################################################
# Lessons Learned
# ---------------
#
# The local step is the most expensive step in terms of runtime (far more costly than the summary or global step)
# Generally, increasing the number of batches has the following effect:
# * Increase the total computational work that must be done for a fixed number of laps
# * Improve the model quality achieved in a limited amount of time, unless the batch size becomes so small that global parameter estimates are poor
#
# We generally recommend considering:
# * batch size around 250 - 2000 (which means set nBatch = nDocsTotal / batch_size)
# * carefully setting the local step convergence threshold (convThrLP could be 0.05 or 0.25 when training, probably needs to be smaller when computing likelihoods for a document)
# * setting the number of iterations per document sufficiently large (might get away with nCoordAscentItersLP = 10 or 25 when training, but might need many iters like 50 or 100 at least when evaluating likelihoods to be confident in the value)

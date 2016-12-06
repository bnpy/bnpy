"""
Functions for estimating topic-by-word matrix using anchor word method

- findAnchorTopics
Provides bnpy-friendly interface to do anchor-word topic recovery.
Calls several customized utilities, like AnchorFinder,
which should be faster and more memory affordable than original impl.

- findAnchorTopics_Orig
Provides same interface to Halpern et al's original functions.

Reference
-------
http://www.cs.nyu.edu/~halpern/code.html
"""

import os
import sys
import numpy as np
import scipy.sparse as sparse
import scipy.io

import AnchorFinder
import RecoverWithL2Loss

import bnpy


def findAnchorTopics(Data, K=10, loss='L2', seed=0,
                     lowerDim=1000, minDocPerWord=0, eps=1e-4, doRecover=1):
    """ Estimate and return K topics using anchor word method

        Modified version of findAnchorTopics_Orig to improve speed, memory.

        Returns
        -------
        topics : 2D array, size K x V, rows sum to one

        References
        -------
        Arora et al. 2013
    """
    # TODO: maybe float32 is cheaper for huge datasets??
    dtype = np.float64
    DWMat = Data.getDocTypeCountMatrix()

    # Select anchors among words that appear in many unique docs
    nDocPerWord = np.sum(DWMat > 0, axis=0)
    candidateRows = np.flatnonzero(nDocPerWord >= minDocPerWord)

    Q = Data.getWordTypeCooccurMatrix(dtype=dtype)
    anchorRows = AnchorFinder.FindAnchorsForSizeKBasis(
        Q, K, candidateRows=candidateRows, seed=seed, lowerDim=lowerDim)

    if doRecover:
        topics = RecoverWithL2Loss.nonNegativeRecoverTopics(
            Q, anchorRows.tolist(), loss, eps=eps)
        assert np.allclose(topics.sum(axis=1), 1.0)
        return topics
    else:
        return Q, anchorRows


def findAnchorTopics_Orig(Data, K=10, loss='L2', seed=0, lowerDim=1000,
                          minDocPerWord=0, eps=1e-4, doRecover=1):
    """ Estimate and return K topics using anchor word method

        Returns
        -------
        topics : numpy 2D array, size K x V

    """
    from Q_matrix import generate_Q_matrix
    from fastRecover import do_recovery

    params = Params(seed=seed, lowerDim=lowerDim,
                    minDocPerWord=minDocPerWord, eps=eps)

    assert isinstance(Data, bnpy.data.DataObj)
    DocWordMat = Data.getSparseDocTypeCountMatrix()

    if not str(type(DocWordMat)).count('csr_matrix') > 0:
        raise NotImplementedError('Need CSR matrix')

    Q = generate_Q_matrix(DocWordMat.copy().T)

    anchors = selectAnchorWords(DocWordMat.tocsc(), Q, K, params)

    if doRecover:
        topics, topic_likelihoods = do_recovery(Q, anchors, loss, params)
        topics = topics.T
        topics = topics / topics.sum(axis=1)[:, np.newaxis]
        return topics
    else:
        return Q, anchors


def selectAnchorWords(DocWordMat, Q, K, params):
    from anchors import findAnchors

    if not str(type(DocWordMat)).count('csc_matrix') > 0:
        raise NotImplementedError('Need CSC matrix')

    nDocsPerWord = np.diff(DocWordMat.indptr)
    candidateWords = np.flatnonzero(nDocsPerWord > params.minDocPerWord)

    anchors = findAnchors(Q, K, params, candidateWords.tolist())
    return anchors


class Params:

    def __init__(self, seed=0, minDocPerWord=10, lowerDim=None, eps=None):
        self.seed = seed
        self.lowerDim = lowerDim
        self.eps = eps
        self.minDocPerWord = minDocPerWord

        self.log_prefix = ""
        self.max_threads = 0

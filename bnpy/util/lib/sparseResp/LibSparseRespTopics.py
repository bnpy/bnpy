'''
LibSparseResp.py

Sets global variable "hasLibReady" with True/False indicator
for whether the compiled cpp library required has compiled and is loadable successfully.
'''
import os
import numpy as np
import ctypes
import scipy.sparse
import sys, sysconfig

from numpy.ctypeslib import ndpointer
from scipy.special import digamma

''' This block of code loads the shared library and defines wrapper functions
    that can take numpy array objects.
'''
libpath = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-1])
libfilename = 'libsparsetopics' + sysconfig.get_config_var('EXT_SUFFIX')
hasLibReady = False

try:
    libpath = os.path.join(libpath, libfilename)
    assert os.path.exists(libpath)
    libTopics = ctypes.cdll.LoadLibrary()
except AssertionError as e:
    hasLibReady = False
except OSError as e:
    # No compiled C++ library exists
    hasLibReady = False
else:
    # libTopics has been loaded correctly from shared library
    hasLibReady = True

    libTopics.sparseLocalStepSingleDoc.restype = None
    libTopics.sparseLocalStepSingleDoc.argtypes = \
        [ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_double,
         ctypes.c_int,
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_int),
         ctypes.c_int,
         ctypes.c_int,
         ndpointer(ctypes.c_int),
         ndpointer(ctypes.c_double),
         ]

    libTopics.sparseLocalStepSingleDoc_ActiveOnly.restype = None
    libTopics.sparseLocalStepSingleDoc_ActiveOnly.argtypes = \
        [ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_double,
         ctypes.c_int,
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_int),
         ctypes.c_int,
         ctypes.c_int,
         ndpointer(ctypes.c_int),
         ndpointer(ctypes.c_double),
         ctypes.c_int,
         ndpointer(ctypes.c_double),
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ndpointer(ctypes.c_int),
         ndpointer(ctypes.c_int),
         ctypes.c_int,
         ]

    libTopics.sparseLocalStepSingleDocWithWordCounts.restype = None
    libTopics.sparseLocalStepSingleDocWithWordCounts.argtypes = \
        [ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_double,
         ctypes.c_int,
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_int),
         ]

def calcSparseLocalParams_SingleDoc(
        wc_d, Lik_d, alphaEbeta, alphaEbetaRem=None,
        topicCount_d_OUT=None,
        spResp_data_OUT=None,
        spResp_colids_OUT=None,
        nCoordAscentItersLP=10, convThrLP=0.001,
        nnzPerRowLP=2,
        restartLP=0,
        restartNumTrialsLP=3,
        activeonlyLP=0,
        initDocTopicCountLP='setDocProbsToEGlobalProbs',
        reviseActiveFirstLP=-1,
        reviseActiveEveryLP=1,
        maxDiffVec=None,
        numIterVec=None,
        nRAcceptVec=None,
        nRTrialVec=None,
        verboseLP=0,
        d=0,
        **kwargs):
    # Parse params for tracking convergence progress
    if maxDiffVec is None:
        maxDiffVec = np.zeros(1, dtype=np.float64)
        numIterVec = np.zeros(1, dtype=np.int32)
    if nRTrialVec is None:
        nRTrialVec = np.zeros(1, dtype=np.int32)
        nRAcceptVec = np.zeros(1, dtype=np.int32)
    assert maxDiffVec.dtype == np.float64
    assert numIterVec.dtype == np.int32
    D = maxDiffVec.size

    N, K = Lik_d.shape
    K1 = alphaEbeta.size
    assert K == K1
    assert topicCount_d_OUT.size == K
    assert spResp_data_OUT.size == N * nnzPerRowLP
    assert spResp_colids_OUT.size == N * nnzPerRowLP
    nnzPerRowLP = np.minimum(nnzPerRowLP, K)
    if initDocTopicCountLP.startswith("fastfirstiter"):
        initProbsToEbeta = -1
    elif initDocTopicCountLP.startswith("setDocProbsToEGlobalProbs"):
        initProbsToEbeta = 1
    else:
        initProbsToEbeta = 0
    if activeonlyLP:
        doTrack = 0
        if reviseActiveFirstLP < 0:
            reviseActiveFirstLP = 2 * nCoordAscentItersLP
        elboVec = np.zeros(doTrack * nCoordAscentItersLP + 1)
        if isinstance(wc_d, np.ndarray) and wc_d.size == N:
            wc_or_allones = wc_d
        else:
            wc_or_allones = np.ones(N)
        libTopics.sparseLocalStepSingleDoc_ActiveOnly(
            Lik_d, wc_or_allones, alphaEbeta,
            nnzPerRowLP, N, K, nCoordAscentItersLP, convThrLP,
            initProbsToEbeta,
            topicCount_d_OUT,
            spResp_data_OUT,
            spResp_colids_OUT,
            d, D, numIterVec, maxDiffVec,
            doTrack, elboVec,
            restartNumTrialsLP * restartLP,
            reviseActiveFirstLP,
            reviseActiveEveryLP,
            nRAcceptVec, nRTrialVec,
            verboseLP,
            )
        if doTrack:
            # Chop off any trailing zeros
            elboVec = elboVec[elboVec != 0.0]
            if elboVec.size > 1 and np.max(np.diff(elboVec)) < -1e-8:
                raise ValueError("NOT MONOTONIC!!!")
    elif isinstance(wc_d, np.ndarray) and wc_d.size == N:
        libTopics.sparseLocalStepSingleDocWithWordCounts(
            wc_d, Lik_d, alphaEbeta,
            nnzPerRowLP, N, K, nCoordAscentItersLP, convThrLP,
            initProbsToEbeta,
            topicCount_d_OUT,
            spResp_data_OUT,
            spResp_colids_OUT,
            )
    else:
        libTopics.sparseLocalStepSingleDoc(
            Lik_d, alphaEbeta,
            nnzPerRowLP, N, K, nCoordAscentItersLP, convThrLP,
            initProbsToEbeta,
            topicCount_d_OUT,
            spResp_data_OUT,
            spResp_colids_OUT,
            d, D, numIterVec, maxDiffVec,
            )


if __name__ == "__main__":
    if not hasLibReady:
        raise NotImplementedError("Library not available.")

    from scipy.special import digamma
    N = 3
    K = 7
    nnzPerRow = 2
    MAXITER = 50
    convThr = 0.005
    alphaEbeta = np.random.rand(K)
    logLik_d = np.log(np.random.rand(N,K) **2)
    wc_d = np.float64(np.arange(1, N+1))
    D = 10
    topicCount_d = np.zeros(K)
    spResp_data = np.zeros(N * D * nnzPerRow)
    spResp_colids = np.zeros(N * D * nnzPerRow, dtype=np.int32)
    for d in [0, 1, 2, 3]:
        print(nnzPerRow)
        start = d * (N * nnzPerRow)
        stop = (d+1) * (N * nnzPerRow)
        libTopics.sparseLocalStepSingleDocWithWordCounts(
            wc_d, logLik_d,
            alphaEbeta,
            nnzPerRow,
            N,
            K,
            MAXITER,
            convThr,
            topicCount_d,
            spResp_data[start:stop],
            spResp_colids[start:stop],
            )
        print(' '.join(['%5.2f' % (x) for x in topicCount_d]))
        print('sum(topicCount_d)=', topicCount_d.sum())
        print('sum(wc_d)=', np.sum(wc_d))
 
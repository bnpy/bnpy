'''
HMMUtil.py

Provides standard message-passing algorithms for inference in HMMs,
such as the forward-backward algorithm

Intentionally separated from rest of HMM code, so that we can swap in
any fast routine for this calculation with ease.
'''
from builtins import *
import numpy as np
from bnpy.util import EPS
from bnpy.util import digamma, gammaln
from bnpy.util.NumericUtil import Config as PlatformConfig
from bnpy.util.NumericUtil import sumRtimesS
from bnpy.util.NumericUtil import inplaceLog
from bnpy.util import as2D

from .lib.LibFwdBwd import cppReady, FwdAlg_cpp, BwdAlg_cpp, SummaryAlg_cpp

def calcLocalParams(Data, LP,
                    transTheta=None, startTheta=None,
                    limitMemoryLP=1,
                    hmm_feature_method_LP='forward+backward',
                    mPairIDs=None,
                    cslice=(0, None),
                    **kwargs):
    ''' Compute local parameters for provided dataset.

    Returns
    -------
    LP : dict of local params, with fields
        * resp : 2D array, nAtom x K
        if limitMemoryLP=0:
            * respPair : 3D array, nAtom x K x K
        if limitMemoryLP=1:
            * TransCount : 3D array, nSeq x K x K
    '''
    # Unpack soft evidence 2D array
    logLik = LP['E_log_soft_ev']
    nAtom, K = logLik.shape

    # Calculate trans prob 2D array
    digammaSumTransTheta = digamma(np.sum(transTheta[:K, :K + 1], axis=1))
    transPi = digamma(transTheta[:K, :K]) - digammaSumTransTheta[:, np.newaxis]
    np.exp(transPi, out=transPi)

    # Calculate LOG of start state prob vector
    logstartPi = digamma(startTheta[:K]) - digamma(np.sum(startTheta[:K + 1]))

    # Set starting probs to uniform,
    # because Line A below updates first state's logLik to include logstartPi
    startPi = np.ones(K)
    logMargPr = np.empty(Data.nDoc)
    resp = np.empty((nAtom, K))

    # Unpack pairs to track for merging.
    if mPairIDs is None:
        mPairIDs = np.zeros((0, 2))
        M = 0
    else:
        if len(mPairIDs) == 0:
            mPairIDs = np.zeros((0, 2))
            M = 0
        else:
            mPairIDs = as2D(mPairIDs)
            M = mPairIDs.shape[0]
    assert mPairIDs.shape[1] == 2
    if hmm_feature_method_LP == 'forward':
        fmsg = np.zeros_like(LP['E_log_soft_ev'])
        # Run forward backward algorithm on each sequence n
        for n in range(Data.nDoc):
            start = Data.doc_range[n]
            stop = Data.doc_range[n + 1]
            logLik_n = logLik[start:stop]
            # Adding in start state probs, in log space for stability.
            logLik_n[0] += logstartPi

            PiInit, PiMat, K = _parseInput_TransParams(startPi, transPi)
            logSoftEv = _parseInput_SoftEv(logLik_n, K)
            T = logSoftEv.shape[0]
            SoftEv, lognormC = expLogLik(logSoftEv)
            fmsg_n, margPrObs = FwdAlg(PiInit, PiMat, SoftEv)
            if not np.all(np.isfinite(margPrObs)):
                raise ValueError('NaN values found. Numerical badness!')
            fmsg[start:stop] = fmsg_n
        LP['fmsg'] = fmsg

    elif limitMemoryLP:
        # Track sufficient statistics directly at each sequence.
        TransCount = np.empty((Data.nDoc, K, K))
        Htable = np.empty((Data.nDoc, K, K))
        mHtable = np.zeros((2 * M, K))

        # Run forward backward algorithm on each sequence n
        for n in range(Data.nDoc):
            start = Data.doc_range[n]
            stop = Data.doc_range[n + 1]
            logLik_n = logLik[start:stop]
            # Adding in start state probs, in log space for stability.
            logLik_n[0] += logstartPi  # Line A

            # Run fwd-fwd alg and record result.
            resp_n, lp_n, TransCount_n, Htable_n, mHtable_n = \
                FwdBwdAlg_LimitMemory(startPi, transPi, logLik_n, mPairIDs)
            resp[start:stop] = resp_n
            logMargPr[n] = lp_n
            TransCount[n] = TransCount_n
            Htable[n] = Htable_n
            mHtable += mHtable_n

        LP['resp'] = resp
        LP['evidence'] = np.sum(logMargPr)
        LP['TransCount'] = TransCount
        LP['Htable'] = Htable
        LP['mHtable'] = mHtable
    else:
        # Track pair-wise assignment probs for each sequence
        respPair = np.empty((nAtom, K, K))

        # Run the forward backward algorithm on each sequence
        for n in range(Data.nDoc):
            start = Data.doc_range[n]
            stop = Data.doc_range[n + 1]
            logLik_n = logLik[start:stop]
            # Adding in start state probs, in log space for stability.
            logLik_n[0] += logstartPi  # Line A

            resp_n, respPair_n, lp_n = \
                FwdBwdAlg(startPi, transPi, logLik_n)
            resp[start:stop] = resp_n
            respPair[start:stop] = respPair_n
            logMargPr[n] = lp_n

        LP['evidence'] = np.sum(logMargPr)
        LP['resp'] = resp
        LP['respPair'] = respPair
    # ... end if statement on limitMemoryLP

    return LP


def FwdBwdAlg(PiInit, PiMat, logSoftEv):
    '''Execute forward-backward algorithm for one sequence.

    Args
    -------
    piInit : 1D array, size K
        initial transition distribution to each of the K states
        must be valid probability vector (positive entries, sums to one)
    piMat  : 2D array, size KxK
        piMat[j] is transition distribution from state j to all K states.
        each row must be probability vector (positive entries, sums to one)
    logSoftEv : 2D array, size TxK
        logSoftEv[t] := log p( x[t] | z[tk] = 1)
                     log likelihood of observation t under state k
                     if given exactly,
                       * resp, respPair will be exact
                       * logMargPrSeq will be exact
                     if given up to an additive constant,
                       * resp, respPair will be exact
                       * logMargPrSeq will be off by an additive constant
    Returns
    -------
    resp : 2D array, size T x K
            resp[t,k] = marg. prob. that step t assigned to state K
                        p( z[t,k] = 1 | x[1], x[2], ... x[T])
    respPair : 3D array, size T x K x K
        respPair[t,j,k] = prob. of the joint event that
        * step t-1 assigned to state j
        * step t assigned to state k
        Formally = p( z[t-1,j] = 1, z[t,k] = 1 | x[1], x[2], ... x[T])
        respPair[0,:,:] is undefined, but kept so indexing consistent.

     logMargPrSeq : scalar real
            logMargPrSeq = joint log probability of the observed sequence
                        log p( x[1], x[2], ... x[T] )
    '''
    PiInit, PiMat, K = _parseInput_TransParams(PiInit, PiMat)
    logSoftEv = _parseInput_SoftEv(logSoftEv, K)
    T = logSoftEv.shape[0]

    SoftEv, lognormC = expLogLik(logSoftEv)

    fmsg, margPrObs = FwdAlg(PiInit, PiMat, SoftEv)
    if not np.all(np.isfinite(margPrObs)):
        raise ValueError('NaN values found. Numerical badness!')

    bmsg = BwdAlg(PiInit, PiMat, SoftEv, margPrObs)
    resp = fmsg * bmsg
    respPair = calcRespPair_fast(PiMat, SoftEv, margPrObs, fmsg, bmsg, K, T)
    logMargPrSeq = np.log(margPrObs).sum() + lognormC.sum()
    return resp, respPair, logMargPrSeq


def FwdBwdAlg_LimitMemory(PiInit, PiMat, logSoftEv, mPairIDs):
    '''Execute forward-backward algorithm using only O(K) memory.

    Args
    -------
    piInit : 1D array, size K
    piMat  : 2D array, size KxK
    logSoftEv : 2D array, size TxK

    Returns
    -------
    resp : 2D array, size T x K
          resp[t,k] = marg. prob. that step t assigned to state K
                      p( z[t,k] = 1 | x[1], x[2], ... x[T])
    TransCount
    Htable
    logMargPrSeq : scalar real
          logMargPrSeq = joint log probability of the observed sequence
                      log p( x[1], x[2], ... x[T] )
    '''
    PiInit, PiMat, K = _parseInput_TransParams(PiInit, PiMat)
    logSoftEv = _parseInput_SoftEv(logSoftEv, K)
    SoftEv, lognormC = expLogLik(logSoftEv)
    T = logSoftEv.shape[0]

    fmsg, margPrObs = FwdAlg(PiInit, PiMat, SoftEv)
    if not np.all(np.isfinite(margPrObs)):
        raise ValueError('NaN values found. Numerical badness!')

    bmsg = BwdAlg(PiInit, PiMat, SoftEv, margPrObs)
    resp = fmsg * bmsg
    logMargPrSeq = np.log(margPrObs).sum() + lognormC.sum()
    TransStateCount, Htable, mHtable = SummaryAlg(
        PiInit, PiMat, SoftEv, margPrObs, fmsg, bmsg, mPairIDs)
    return resp, logMargPrSeq, TransStateCount, Htable, mHtable


def calcRespPair_forloop(PiMat, SoftEv, margPrObs, fmsg, bmsg, K, T):
    ''' Calculate pair-wise responsibilities for all adjacent timesteps

    Uses a simple, for-loop implementation.
    See calcRespPair_fast for a equivalent function that is much faster.


    Returns
    ---------
    respPair : 3D array, size T x K x K
        respPair[t,j,k] = prob. of the joint event that
        * step t-1 assigned to state j
        * step t assigned to state k
        Formally = p( z[t-1,j] = 1, z[t,k] = 1 | x[1], x[2], ... x[T])
        respPair[0,:,:] is undefined, but kept so indexing consistent.
    '''
    respPair = np.zeros((T, K, K))
    for t in range(1, T):
        respPair[t] = np.outer(fmsg[t - 1], bmsg[t] * SoftEv[t])
        respPair[t] *= PiMat / margPrObs[t]
    return respPair


def calcRespPair_fast(PiMat, SoftEv, margPrObs, fmsg, bmsg, K, T,
                      doCopy=0):
    ''' Calculate pair-wise responsibilities for all adjacent timesteps

    Uses a fast, vectorized algorithm.

    Returns
    ---------
    respPair : 3D array, size T x K x K
        respPair[t,j,k] = prob. of the joint event that
        * step t-1 assigned to state j
        * step t assigned to state k
        Formally = p( z[t-1,j] = 1, z[t,k] = 1 | x[1], x[2], ... x[T])
        respPair[0,:,:] is undefined, but kept so indexing consistent.
    '''
    if doCopy:
        bmsgSoftEv = SoftEv * bmsg
    else:
        bmsgSoftEv = SoftEv  # alias
        bmsgSoftEv *= bmsg  # in-place multiplication

    respPair = np.zeros((T, K, K))
    respPair[1:] = fmsg[:-1][:, :, np.newaxis] * \
        bmsgSoftEv[1:][:, np.newaxis, :]
    respPair *= PiMat[np.newaxis, :, :]
    respPair /= margPrObs[:, np.newaxis, np.newaxis]
    return respPair


def FwdAlg(PiInit, PiMat, SoftEv):
    ''' Forward algorithm for a single HMM sequence. Wrapper for py/cpp.

    Related
    -------
    FwdAlg_py

    Returns
    -------
    fmsg : 2D array, size T x K
        fmsg[t,k] = p( z[t,k] = 1 | x[1] ... x[t] )
    margPrObs : 1D array, size T
        margPrObs[t] = p( x[t] | x[1], x[2], ... x[t-1] )
    '''
    if cppReady() and PlatformConfig['FwdBwdImpl'] == "cpp":
        return FwdAlg_cpp(PiInit, PiMat, SoftEv)
    else:
        return FwdAlg_py(PiInit, PiMat, SoftEv)


def BwdAlg(PiInit, PiMat, SoftEv, margPrObs):
    ''' Backward algorithm for a single HMM sequence.

    Wrapper for BwdAlg_py/BwdAlg_cpp.

    Related
    -------
    BwdAlg_py

    Returns
    -------
    bmsg : 2D array, size TxK
            bmsg[t,k] = p( x[t+1], x[t+2], ... x[T] |  z[t,k] = 1 )
                        -------------------------------------
                        p( x[t+1], x[t+2], ... x[T] |  x[1] ... x[t])
    '''
    if cppReady() and PlatformConfig['FwdBwdImpl'] == "cpp":
        return BwdAlg_cpp(PiInit, PiMat, SoftEv, margPrObs)
    else:
        return BwdAlg_py(PiInit, PiMat, SoftEv, margPrObs)


def FwdAlg_py(PiInit, PiMat, SoftEv):
    ''' Forward algorithm for a single HMM sequence. In pure python.

    Execute forward message-passing on an observed sequence
    given HMM state transition params and likelihoods of each observation

    Args
    -------
    piInit : 1D array, size K
      initial transition distribution to each of the K states
      must be valid probability vector (positive entries, sums to one)
    piMat  : 2D array, size KxK
      piMat[j] is transition distribution from state j to all K states.
      each row must be probability vector (positive entries, sums to one)
    SoftEv : 2D array, size TxK
      SoftEv[t] := p( x[t] | z[tk] = 1)
                   likelihood of observation t under state k
                   given up to an additive constant for each t
    Returns
    -------
    fmsg : 2D array, size T x K
        fmsg[t,k] = p( z[t,k] = 1 | x[1] ... x[t] )
    margPrObs : 1D array, size T
        margPrObs[t] = p( x[t] | x[1], x[2], ... x[t-1] )
    '''
    T = SoftEv.shape[0]
    K = PiInit.size
    PiTMat = PiMat.T

    fmsg = np.empty((T, K))
    margPrObs = np.zeros(T)
    for t in range(0, T):
        if t == 0:
            fmsg[t] = PiInit * SoftEv[0]
        else:
            fmsg[t] = np.dot(PiTMat, fmsg[t - 1]) * SoftEv[t]
        margPrObs[t] = np.sum(fmsg[t])
        fmsg[t] /= margPrObs[t]
    return fmsg, margPrObs


def BwdAlg_py(PiInit, PiMat, SoftEv, margPrObs):
    '''Backward algorithm for a single HMM sequence. In pure python.

    Takes as input the HMM state transition params,
    initial probabilities, and likelihoods of each observation.
    Requires running forward filtering first, to obtain correct scaling.

    Args
    -------
    piInit : 1D array, size K
        initial transition distribution to each of the K states
        must be valid probability vector (positive entries, sums to one)
    piMat  : 2D array, size KxK
        piMat[j] is transition distribution from state j to all K states.
        each row must be probability vector (positive entries, sums to one)
    SoftEv : 2D array, size TxK
        SoftEv[t] := p( x[t] | z[tk] = 1)
            likelihood of observation t under state k
            given up to an additive constant for each t
    margPrObs : 1D array, size T
        margPrObs[t] := p( x[t] | x[1], x[2], ... x[t-1] )
        this is returned by FwdAlg

    Returns
    -------
    bmsg : 2D array, size TxK
        bmsg[t,k] = p( x[t+1], x[t+2], ... x[T] |  z[t,k] = 1 )
                    -------------------------------------
                    p( x[t+1], x[t+2], ... x[T] |  x[1] ... x[t])
    '''
    T = SoftEv.shape[0]
    K = PiInit.size
    bmsg = np.ones((T, K))
    for t in range(T - 2, -1, -1):
        bmsg[t] = np.dot(PiMat, bmsg[t + 1] * SoftEv[t + 1])
        bmsg[t] /= margPrObs[t + 1]
    return bmsg


def SummaryAlg(*args):
    ''' Summarize pairwise potentials of single HMM sequence.

    Related
    -------
    SummaryAlg_py

    Returns
    -------
    TransStateCount
    Htable
    '''
    if cppReady() and PlatformConfig['FwdBwdImpl'] == "cpp":
        return SummaryAlg_cpp(*args)
    else:
        return SummaryAlg_py(*args)


def SummaryAlg_py(PiInit, PiMat, SoftEv, margPrObs, fMsg, bMsg,
                  mPairIDs=None):
    K = PiInit.size
    T = SoftEv.shape[0]
    if mPairIDs is None:
        M = 0
    else:
        if len(mPairIDs) == 0:
            M = 0
        else:
            mPairIDs = as2D(np.asarray(mPairIDs, dtype=np.int32))
            assert mPairIDs.ndim == 2
            assert mPairIDs.shape[1] == 2
            assert mPairIDs.shape[0] > 0
            M = mPairIDs.shape[0]
    mHtable = np.zeros((2 * M, K))

    respPair_t = np.zeros((K, K))
    Htable = np.zeros((K, K))
    TransStateCount = np.zeros((K, K))
    for t in range(1, T):
        respPair_t = np.outer(fMsg[t - 1], bMsg[t] * SoftEv[t])
        respPair_t *= PiMat / margPrObs[t]
        TransStateCount += respPair_t

        respPair_t += 1e-100
        rowwiseSum = np.sum(respPair_t, axis=1)
        Htable += respPair_t * np.log(respPair_t) \
            - respPair_t * np.log(rowwiseSum)[:, np.newaxis]

    if M > 0:
        respPair = calcRespPair_fast(PiMat, SoftEv,
                                     margPrObs, fMsg, bMsg,
                                     K, T, doCopy=1)
        for m in range(M):
            kA = mPairIDs[m, 0]
            kB = mPairIDs[m, 1]
            mHtable[
                2 *
                m:2 *
                m +
                2] = calc_sub_Htable_forMergePair(
                respPair,
                kA,
                kB)

    Htable *= -1
    return TransStateCount, Htable, mHtable


def expLogLik(logSoftEv, axis=1):
    ''' Return element-wise exp of input log likelihood

    Numerically safe, guaranteed not to underflow

    Returns
    --------
    SoftEv : 2D array, size TxK
              equal to exp(logSoftEv), up to prop constant for each row
    lognormC : 1D array, size T
              gives log of the prop constant for each row
    '''
    lognormC = np.max(logSoftEv, axis)
    if axis == 0:
        logSoftEv = logSoftEv - lognormC[np.newaxis, :]
    elif axis == 1:
        logSoftEv = logSoftEv - lognormC[:, np.newaxis]
    SoftEv = np.exp(logSoftEv)
    return SoftEv, lognormC


def _parseInput_TransParams(PiInit, PiMat):
    PiInit = np.asarray(PiInit, dtype=np.float64)
    PiMat = np.asarray(PiMat, dtype=np.float64)
    assert PiInit.ndim == 1
    K0 = PiInit.shape[0]
    assert PiMat.ndim == 2
    J, K = PiMat.shape
    assert J == K
    assert K0 == K
    return PiInit, PiMat, K


def _parseInput_SoftEv(logSoftEv, K):
    logSoftEv = np.asarray(logSoftEv, dtype=np.float64)
    Tl, Kl = logSoftEv.shape
    assert Kl == K
    return logSoftEv


def runViterbiAlg(logSoftEv, logPi0, logPi):
    ''' Run viterbi algorithm to estimate MAP states for single sequence.

    Args
    ------
    logSoftEv : 2D array, T x K
        log soft evidence matrix
        each row t := log p( x[t] | z[t]=k )
    pi0 : 1D array, length K
        initial state probability vector, sums to one
    pi : 2D array, shape K x K
        j-th row is is transition probability vector for state j

    Returns
    ------
    zHat : 1D array, length T, representing the MAP state sequence
        zHat[t] gives the integer label {1, 2, ... K} of state at timestep t
    '''
    if np.any(logPi0 > 0):
        logPi0 = np.log(logPi0 + EPS)
    if np.any(logPi > 0):
        logPi = np.log(logPi + EPS)
    T, K = np.shape(logSoftEv)

    # ScoreTable : 2D array, shape T x K
    #   entry t,k gives the log probability of reaching state k at time t
    #   under the most likely path
    ScoreTable = np.zeros((T, K))

    # PtrTable : 2D array, size T x K
    #   entry t,k gives the integer id of the state j at timestep t-1
    #   which would be part of the most likely path to reaching k at t
    PtrTable = np.zeros((T, K))

    ScoreTable[0, :] = logSoftEv[0] + logPi0
    PtrTable[0, :] = -1

    ids0toK = list(range(K))
    for t in range(1, T):
        ScoreMat_t = logPi + ScoreTable[t - 1, :][:, np.newaxis]
        bestIDvec = np.argmax(ScoreMat_t, axis=0)

        PtrTable[t, :] = bestIDvec
        ScoreTable[t, :] = logSoftEv[t, :] \
            + ScoreMat_t[(bestIDvec, ids0toK)]

    # Follow backward pointers to construct most likely state sequence
    z = np.zeros(T)
    z[-1] = np.argmax(ScoreTable[-1])
    for t in reversed(range(T - 1)):
        z[t] = PtrTable[t + 1, int(z[t + 1])]
    return z


def runViterbiAlg_forloop(logSoftEv, logPi0, logPi):
    ''' Run viterbi algorithm to estimate MAP states for single sequence.

    This method will produce the same output as runViterbiAlg,
    but will be much simpler to read, since it uses an inner for-loop
    instead of complete vectorization
    '''
    if np.any(logPi0 > 0):
        logPi0 = np.log(logPi0 + EPS)
    if np.any(logPi > 0):
        logPi = np.log(logPi + EPS)
    T, K = np.shape(logSoftEv)

    # ScoreTable : 2D array, shape T x K
    #   entry t,k gives the log probability of reaching state k at time t
    #   under the most likely path
    ScoreTable = np.zeros((T, K))

    # PtrTable : 2D array, size T x K
    #   entry t,k gives the integer id of the state j at timestep t-1
    #   which would be part of the most likely path to reaching k at t
    PtrTable = np.zeros((T, K))

    ScoreTable[0, :] = logSoftEv[0] + logPi0
    PtrTable[0, :] = -1

    for t in range(1, T):
        for k in range(K):
            ScoreVec = logPi[:, k] + ScoreTable[t - 1, :]
            bestID = np.argmax(ScoreVec)

            PtrTable[t, k] = bestID
            ScoreTable[t, k] = logSoftEv[t, k] + ScoreVec[bestID]

    # Follow backward pointers to construct most likely state sequence
    z = np.zeros(T)
    z[-1] = np.argmax(ScoreTable[-1])
    for t in reversed(range(T - 1)):
        z[t] = PtrTable[t + 1, z[t + 1]]
    return z


def calcEntropyFromResp_fast(resp, respPair,
                             Data=None, startLocIDs=None, eps=1e-100):
    ''' Calculate state assignment entropy for all sequences.

        Fast, vectorized. Purely numpy.

        Returns
        --------
        H : positive scalar
    '''
    if startLocIDs is not None:
        startLocIDs = np.asarray(startLocIDs)

    if Data is not None:
        startLocIDs = Data.doc_range[:-1]

    sigma = respPair / (respPair.sum(axis=2)[:, :, np.newaxis] + eps)
    firstH = -1 * np.sum(resp[startLocIDs] * np.log(resp[startLocIDs] + eps))
    restH = -1 * np.sum(respPair[1:, :, :] * np.log(sigma[1:, :, :] + eps))
    return firstH + restH


def calcEntropyFromResp_faster(resp, respPair,
                               Data=None, startLocIDs=None, eps=1e-100):
    ''' Calculate state assignment entropy for all sequences.

    Fast, vectorized. Can use numexpr to speed up computation if available.

    Returns
    --------
    H : positive scalar
    '''
    if startLocIDs is not None:
        startLocIDs = np.asarray(startLocIDs)

    if Data is not None:
        startLocIDs = Data.doc_range[:-1]

    firstH = -1 * np.sum(resp[startLocIDs] * np.log(resp[startLocIDs] + eps))
    restH = calc_Htable(respPair).sum()
    return firstH + restH


def calcEntropyFromResp_forloop(resp, respPair, Data, eps=1e-100):
    ''' Calculate state assignment entropy for all sequences. Using forloop.

    Exactly same input/output as calcEntropyFromResp, just with
    easier to read implementation. Useful for verifying correctness.
    '''
    totalH = 0
    for n in range(Data.nDoc):
        start = Data.doc_range[n]
        stop = Data.doc_range[n + 1]
        resp_n = resp[start:stop]
        respPair_n = respPair[start:stop]

        # sigma_n : conditional prob of each adjacent pair of states
        # sums to one over the final dimension: sigma_n[t, j, :].sum()
        sigma_n = respPair_n / (respPair_n.sum(axis=2)[:, :, np.newaxis] + eps)

        # Entropy of the first step t=1
        firstH_n = -1 * np.inner(resp_n[0], np.log(resp_n[0] + eps))

        # Entropy of the remaining steps 2, 3, ... T
        restH_n = -1 * np.sum(respPair_n * np.log(sigma_n + eps))
        totalH += firstH_n + restH_n
    return totalH

# Defining this directly should avoid overhead of extra function call
calcEntropyFromResp = calcEntropyFromResp_faster


def PrecompMergeEntropy_SpecificPairs(LP, Data, mPairIDs, eps=1e-100):
    ''' Calculate replacement tables for specific candidate merge pairs

    Returns
    --------
    sub_Hstart : 1D array, size M
    sub_Htable : 3D array, size M x 2 x K
    '''
    resp = LP['resp']
    startLocIDs = Data.doc_range[:-1]
    K = resp.shape[1]

    sub_Hstart = np.zeros(len(mPairIDs))
    sub_Htable = np.zeros((len(mPairIDs), 2, K))

    for mID, mPair in enumerate(mPairIDs):
        kA, kB = mPair
        sub_Hstart[mID] = calc_sub_Hstart_forMergePair(
            resp, kA, kB, Data=Data, eps=eps)
        if 'mHtable' in LP:
            sub_Htable[mID] = LP['mHtable'][(2 * mID):(2 * mID + 2)]
        else:
            sub_Htable[mID] = calc_sub_Htable_forMergePair(
                LP['respPair'], kA, kB, eps=eps)
    return sub_Hstart, sub_Htable


def calc_sub_Htable_forMergePair(respPair, kA, kB,
                                 rowSums=None, eps=1e-100):
    ''' Calculate replacement entries of Htable for specific candidate merge pair

    Returns
    --------
    mergeH : 2D array, shape 2 x K
             mergeH[0, :] gives values to replace along row kA
             mergeH[1, :] gives values to replace along col kA
    '''
    K = respPair.shape[1]
    mergeH = np.zeros((2, K))

    if rowSums is None:
        rowSums = np.sum(respPair, axis=2) + eps

    # Calculate new "outgoing" (row) entropy terms for the new state
    mr_resp = respPair[:, kA, :] + respPair[:, kB, :]
    mr_sigm = mr_resp / (rowSums[:, kA] + rowSums[:, kB])[:, np.newaxis]
    mergeH[0, :] = -1 * np.sum(mr_resp * np.log(mr_sigm + 1e-100), axis=0)

    # Calculate new "incoming" (col) entropy terms for the new state
    mc_resp = respPair[:, :, kA] + respPair[:, :, kB]
    mc_sigm = mc_resp / rowSums
    mergeH[1, :] = -1 * np.sum(mc_resp * np.log(mc_sigm + 1e-100), axis=0)

    # Calculate special term for intersection of kA/kB
    mi_resp = respPair[:, kA, kA] + respPair[:, kB, kB] \
        + respPair[:, kA, kB] + respPair[:, kB, kA]
    mi_sigm = mi_resp / (rowSums[:, kA] + rowSums[:, kB])
    mergeH[:, kA] = -1 * np.sum(mi_resp * np.log(mi_sigm + 1e-100))
    mergeH[:, kB] = 0
    return mergeH


def calc_sumHtable_forMergePair__fromResp(respPair, kA, kB,
                                          rowSums=None, eps=1e-100):
    ''' Calculate sum of Htable matrix after merger of specific pair of states.

    Directly compute this sum using the local parameters respPair.

    Useful as a test to verify that the more efficient method is correct.

    Returns
    --------
    L_entropy : scalar
                exact value of entropy term for candidate
    '''
    m_respPair = np.delete(respPair, kB, axis=1)
    m_respPair = np.delete(m_respPair, kB, axis=2)

    # Fill in new state's column
    m_respPair[:, :kB, kA] += respPair[:, :kB, kB]
    m_respPair[:, kB:, kA] += respPair[:, kB + 1:, kB]

    # Fill in new state's rows
    m_respPair[:, kA, :kB] += respPair[:, kB, :kB]
    m_respPair[:, kA, kB:] += respPair[:, kB, kB + 1:]

    # Fill in new state's self-trans values
    m_respPair[:, kA, kA] = respPair[:, kA, kA] + \
        respPair[:, kA, kB] + \
        respPair[:, kB, kB] + \
        respPair[:, kB, kA]
    assert np.allclose(1.0, m_respPair[1:].sum(axis=2).sum(axis=1))

    # Now, do the exact entropy calculation
    m_sigma = m_respPair / (m_respPair.sum(axis=2)[:, :, np.newaxis] + eps)
    m_H = -1 * np.sum(m_respPair * np.log(m_sigma + eps), axis=0)
    return m_H.sum()


def calc_sumHtable_forMergePair__fromTables(Htable_orig, Mtable, kA, kB):
    ''' Calculate sum of Htable matrix after merger of specific pair of states.

    Use precomputed tables, rather than local parameters.
    Only consider the non-starting-state entries.

    Returns
    --------
    L_entropy : scalar
                exact value of entropy term for candidate
    '''
    Htable_new = calc_Htable_forMergePair_fromTables(
        Htable_orig, Mtable, kA, kB)
    return Htable_new.sum()


def calc_Htable_forMergePair_fromTables(Htable_orig, Mtable, kA, kB):
    ''' Calculate Htable matrix after merger of specific pair of states

    Use precomputed tables, never touch any local parameters
    Only consider the non-starting-state entries.

    Returns
    --------
    Htable : 2D array, size K-1 x K-1
                exact value of entropy for each state pair for candidate
    '''
    assert kA < kB
    assert Mtable.shape[0] == 2

    Htable_new = Htable_orig.copy()
    Htable_new[kA, :] = Mtable[0]
    Htable_new[:, kA] = Mtable[1]
    Htable_new = np.delete(Htable_new, kB, axis=1)
    Htable_new = np.delete(Htable_new, kB, axis=0)
    return Htable_new


def calc_Htable(respPair, eps=1e-100):
    ''' Calculate table of state assignment entropy for all sequences.

    Fast, vectorized.

    Returns
    --------
    Htable : 2D array, K x K
             sum of the entries yields total entropy
    '''
    sigma = respPair / (respPair.sum(axis=2)[:, :, np.newaxis] + eps)
    sigma += eps  # make it safe for taking logs!
    logsigma = sigma  # alias
    inplaceLog(logsigma)  # use fast numexpr library if possible
    H_KxK = -1 * sumRtimesS(respPair, logsigma)
    return H_KxK


def calc_Hstart(resp, Data=None, startLocIDs=None,
                eps=1e-100):
    ''' Calculate vector of start-state entropy for all sequences.

    Returns
    --------
    Hstart : 1D array, size K
             Hstart[k] = -1 * \sum_{n=1}^N r_{n1k} log r_{n1k}
    '''
    if startLocIDs is not None:
        startLocIDs = np.asarray(startLocIDs)
    if Data is not None:
        startLocIDs = Data.doc_range[:-1]

    startresp = resp[startLocIDs]
    firstHvec = -1 * np.sum(startresp * np.log(startresp + eps), axis=0)
    return firstHvec


def calc_sub_Hstart_forMergePair(resp, kA, kB,
                                 Data=None, startLocIDs=None,
                                 eps=1e-100):
    ''' Calculate Hstart value for specific merge pair.

    This value will be substituted into Hstart vector to calculate total
    entropy of the starting state.

    Returns
    --------
    Hstart : scalar
    '''
    if startLocIDs is not None:
        startLocIDs = np.asarray(startLocIDs)
    if Data is not None:
        startLocIDs = Data.doc_range[:-1]

    startresp = resp[startLocIDs, kA] + resp[startLocIDs, kB]
    return -1 * np.sum(startresp * np.log(startresp + eps), axis=0)


def construct_LP_forMergePair(Data, LP, kA, kB):
    ''' Create new local param (LP) for a merge of states kA, kB
    '''
    Tall = LP['resp'].shape[0]
    K = LP['resp'].shape[-1]

    m_resp = np.zeros((Tall, K - 1))
    m_respPair = np.zeros((Tall, K - 1, K - 1))
    for n in range(Data.nDoc):
        start = Data.doc_range[n]
        stop = Data.doc_range[n + 1]

        # Make respPair for candidate
        assert np.allclose(LP['respPair'][start].sum(), 0.0)
        for t in range(start + 1, stop):
            m_respPair[t] = mergeKxK_forSinglePair(LP['respPair'][t], kA, kB)

        # Make resp for candidate
        for k in range(K):
            if k == kA:
                m_resp[start:stop, k] = LP['resp'][start:stop, kA] \
                    + LP['resp'][start:stop, kB]
            elif k == kB:
                continue
            elif k > kB:
                m_resp[start:stop, k - 1] = LP['resp'][start:stop, k]
            elif k < kB:
                m_resp[start:stop, k] = LP['resp'][start:stop, k]

        assert np.allclose(1.0, m_resp[start:stop].sum(axis=1))
        assert np.allclose(
            1.0,
            m_respPair[
                start +
                1:stop].sum(
                axis=2).sum(
                axis=1))

    # Return
    assert np.allclose(m_resp.sum(), m_respPair.sum() + Data.nDoc)
    return dict(resp=m_resp, respPair=m_respPair)


def mergeKxK_forSinglePair(X, kA, kB):
    Y = X.copy()
    Y[:, kA] += Y[:, kB]
    Y[kA, :] += Y[kB, :]
    Y = np.delete(Y, kB, axis=1)
    Y = np.delete(Y, kB, axis=0)
    return Y

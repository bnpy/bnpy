'''
MergePlanner.py

Contains methods necessary for advanced selection of which components to merge.
'''
import numpy as np
from collections import defaultdict

from bnpy.util import isEvenlyDivisibleFloat
import bnpy.mergemove.MergeLogger as MergeLogger

# Constant defining how far calculated ELBO gap can be from zero
# and still be considered accepted or favorable
from bnpy.mergemove.MergeMove import ELBO_GAP_ACCEPT_TOL

CountTracker = defaultdict(int)


def preselectPairs(curModel, SS, lapFrac,
                   mergePairSelection='wholeELBO',
                   prevScoreMat=None,
                   mergeScoreRefreshInterval=10,
                   mergeMaxDegree=5, **kwargs):
    ''' Create list of candidate pairs for merge
    '''
    needRefresh = isEvenlyDivisibleFloat(lapFrac, mergeScoreRefreshInterval)
    if prevScoreMat is None or needRefresh:
        ScoreMat = np.zeros((SS.K, SS.K))
        doAllPairs = 1
    else:
        assert prevScoreMat.shape[0] == SS.K
        ScoreMat = prevScoreMat
        doAllPairs = 0
    ScoreMat = updateScoreMat_wholeELBO(ScoreMat, curModel, SS, doAllPairs)

    posMask = ScoreMat > - ELBO_GAP_ACCEPT_TOL
    Nvec = SS.getCountVec()
    tinyVec = Nvec < 25
    tinyMask = np.add(tinyVec, tinyVec[:, np.newaxis])
    posAndTiny = np.logical_and(posMask, tinyMask)
    posAndBothBig = np.logical_and(posMask, 1 - tinyMask)

    # Select list of pairs to track for merge
    # prioritizes merges that make big changes
    # avoids tracking too many pairs that involves same node
    pairsBig = selectPairsUsingAtMostNOfEachComp(posAndBothBig,
                                                 N=mergeMaxDegree)
    scoresBig = np.asarray([ScoreMat[a, b] for (a, b) in pairsBig])
    pairsBig = [pairsBig[x] for x in np.argsort(-1 * scoresBig)]

    pairsTiny = selectPairsUsingAtMostNOfEachComp(posAndTiny, pairsBig,
                                                  N=mergeMaxDegree,
                                                  Nextra=2)
    scoresTiny = np.asarray([ScoreMat[a, b] for (a, b) in pairsTiny])
    pairsTiny = [pairsTiny[x] for x in np.argsort(-1 * scoresTiny)]
    return pairsBig + pairsTiny, ScoreMat


def calcDegreeFromEdgeList(pairIDs, nNode):
    ''' Calculate degree of each node given edge list

        Returns
        -------
        degree : 1D array, size nNode
        degree[k] counts number of edges that node k appears in
    '''
    degree = np.zeros(nNode, dtype=np.int32)
    for n in range(nNode):
        degree[n] = np.sum([n in pair for pair in pairIDs])
    return degree


def selectPairsUsingAtMostNOfEachComp(AdjMat, extraFixedEdges=None,
                                      N=3, Nextra=0):
    '''
        Args
        --------
        AdjMat : 2D array, size K x K
        N : max degree of each node

        Returns
        --------
        pairIDs : list of tuples, one entry per selected pair
    '''
    if np.sum(AdjMat) == 0:
        return list()

    # AMat :
    # tracks all remaining CANDIDATE edges where both node under the degree
    # limit.
    AMat = AdjMat.copy()

    xdegree = np.zeros(AdjMat.shape[0], dtype=np.int32)
    if extraFixedEdges is not None:
        for kA, kB in extraFixedEdges:
            xdegree[kA] += 1
            xdegree[kB] += 1

    # degree : tracks CANDIDATE edges (including extra) that not excluded
    # newdegree : tracks edges we will KEEP
    newdegree = np.zeros_like(xdegree)
    newdegree += xdegree

    exhaustedMask = newdegree >= N
    AMat[exhaustedMask, :] = 0
    AMat[:, exhaustedMask] = 0
    degree = np.sum(AMat, axis=0) + np.sum(AMat, axis=1) + xdegree

    # Traverse comps from largest to smallest degree
    pairIDs = list()
    nodeOrder = np.argsort(-1 * degree)
    for nodeID in nodeOrder:

        # Get list of remaining possible partners for node
        partners = np.flatnonzero(AMat[nodeID, :] + AMat[:, nodeID])

        # Sort node's partners from smallest to largest degree,
        # since we want to prioritize keeping small degree partners
        partners = partners[np.argsort([degree[p] for p in partners])]

        Ncur = N - newdegree[nodeID]
        keepPartners = partners[:Ncur]
        rejectPartners = partners[Ncur:]

        for p in keepPartners:
            kA = np.minimum(p, nodeID)
            kB = np.maximum(p, nodeID)
            pairIDs.append((kA, kB))
            AMat[kA, kB] = 0  # make pair ineligible for future partnerships
            newdegree[p] += 1
            newdegree[nodeID] += 1

        for p in rejectPartners:
            kA = np.minimum(p, nodeID)
            kB = np.maximum(p, nodeID)
            AMat[kA, kB] = 0  # make pair ineligible for future partnerships
            degree[p] -= 1
            degree[nodeID] -= 1

        exhaustedMask = newdegree >= N
        AMat[exhaustedMask, :] = 0
        AMat[:, exhaustedMask] = 0
        degree = np.sum(AMat, axis=0) + np.sum(AMat, axis=1) + xdegree

    cond1 = np.allclose(degree, xdegree)
    cond2 = np.max(newdegree) <= N + Nextra
    if not cond1:
        print('WARNING: BAD DEGREE CALCULATION')
    if not cond2:
        print('WARNING: BAD NEWDEGREE CALCULATION')
        print('max(newdegree)=%d' % (np.max(newdegree)))
        print('N + Nextra: %d' % (N + Nextra))
    return pairIDs


def updateScoreMat_wholeELBO(ScoreMat, curModel, SS, doAllPairs=0):
    ''' Calculate upper-tri matrix of exact ELBO gap for each candidate pair

        Returns
        ---------
        Mraw : 2D array, size K x K. Uppert tri entries carry content.
            Mraw[j,k] gives the scalar ELBO gap for the potential merge of j,k
    '''
    K = SS.K
    if doAllPairs:
        AGap = curModel.allocModel.calcHardMergeGap_AllPairs(SS)
        OGap = curModel.obsModel.calcHardMergeGap_AllPairs(SS)
        ScoreMat = AGap + OGap
        ScoreMat[np.tril_indices(SS.K)] = -np.inf
        for k, uID in enumerate(SS.uIDs):
            CountTracker[uID] = SS.getCountVec()[k]
        nUpdated = SS.K * (SS.K - 1) / 2
    else:
        ScoreMat[np.tril_indices(SS.K)] = -np.inf
        # Rescore only specific pairs that are positive
        redoMask = ScoreMat > -1 * ELBO_GAP_ACCEPT_TOL
        for k, uID in enumerate(SS.uIDs):
            if CountTracker[uID] == 0:
                # Always precompute for brand-new comps
                redoMask[k, :] = 1
                redoMask[:, k] = 1
            else:
                absDiff = np.abs(SS.getCountVec()[k] - CountTracker[uID])
                percDiff = absDiff / (CountTracker[uID] + 1e-10)
                if percDiff > 0.25:
                    redoMask[k, :] = 1
                    redoMask[:, k] = 1
                    CountTracker[uID] = SS.getCountVec()[k]
        redoMask[np.tril_indices(SS.K)] = 0
        aList, bList = np.unravel_index(np.flatnonzero(redoMask), (SS.K, SS.K))

        if len(aList) > 0:
            mPairIDs = list(zip(aList, bList))
            AGap = curModel.allocModel.calcHardMergeGap_SpecificPairs(
                SS, mPairIDs)
            OGap = curModel.obsModel.calcHardMergeGap_SpecificPairs(
                SS, mPairIDs)
            ScoreMat[aList, bList] = AGap + OGap
        nUpdated = len(aList)
    MergeLogger.log('MERGE ScoreMat Updates: %d entries.' % (nUpdated),
                    level='debug')
    return ScoreMat


def preselect_candidate_pairs(curModel, SS,
                              randstate=np.random.RandomState(0),
                              mergePairSelection='random',
                              mergePerLap=10,
                              doLimitNumPairs=1,
                              M=None,
                              **kwargs):
    ''' Get a list of tuples representing candidate pairs to merge.

    Args
    --------
    curModel : bnpy HModel
    SS : bnpy SuffStatBag. If None, defaults to random selection.
    randstate : numpy random number generator
    mergePairSelection : name of procedure to select candidate pairs
    mergePerLap : int number of candidates to identify
                   (may be less if K small)

    Returns
    --------
    mPairList : list of tuples
        each entry is a tuple of two integers
        indicating component ID candidates for positions kA, kB
    '''
    kwargs['mergePairSelection'] = mergePairSelection
    kwargs['randstate'] = randstate
    if 'excludePairs' not in kwargs:
        excludePairs = list()
    else:
        excludePairs = kwargs['excludePairs']

    K = curModel.allocModel.K
    if doLimitNumPairs:
        nMergeTrials = mergePerLap + kwargs['mergeNumExtraCandidates']
    else:
        nMergeTrials = K * (K - 1) // 2

    if SS is None:  # Handle first lap
        kwargs['mergePairSelection'] = 'random'

    Mraw = None

    # Score matrix
    # M : 2D array, shape K x K
    #     M[j,k] = score for viability of j,k.  Larger = better.
    selectroutine = kwargs['mergePairSelection']
    if kwargs['mergePairSelection'].count('random') > 0:
        M = kwargs['randstate'].rand(K, K)
    elif kwargs['mergePairSelection'].count('marglik') > 0:
        M = calcScoreMatrix_marglik(curModel, SS, excludePairs)
    elif kwargs['mergePairSelection'].count('wholeELBO') > 0:
        M, Mraw = calcScoreMatrix_wholeELBO(curModel, SS, excludePairs, M=M)
    elif kwargs['mergePairSelection'].count('corr') > 0:
        # Use correlation matrix as score for selecting candidates!
        if selectroutine.count('empty') > 0:
            M = calcScoreMatrix_corrOrEmpty(SS)
        elif selectroutine.count('degree') > 0:
            M = calcScoreMatrix_corrLimitDegree(SS)
        else:
            M = calcScoreMatrix_corr(SS)
    else:
        raise NotImplementedError(kwargs['mergePairSelection'])

    # Only upper-triangular indices are allowed.
    M[np.tril_indices(K)] = 0
    # Excluded pairs are not allowed.
    M[list(zip(*excludePairs))] = 0

    # Select candidates
    aList, bList = _scorematrix2rankedlist_greedy(M, nMergeTrials)

    # Return completed lists
    assert len(aList) == len(bList)
    assert len(aList) <= nMergeTrials
    assert len(aList) <= K * (K - 1) // 2
    assert np.all(np.asarray(aList) < np.asarray(bList))

    if 'returnScoreMatrix' in kwargs and kwargs['returnScoreMatrix']:
        if Mraw is None:
            return list(zip(aList, bList)), M
        else:
            return list(zip(aList, bList)), Mraw
    return list(zip(aList, bList))


def _scorematrix2rankedlist_greedy(M, nPairs, doKeepZeros=False):
    ''' Return the nPairs highest-ranked pairs in score matrix M

        Args
        -------
          M : score matrix, K x K
              should have only entries kA,kB where kA <= kB

        Returns
        --------
          aList : list of integer ids for rows of M
          bList : list of integer ids for cols of M

        Example
        ---------
        _scorematrix2rankedlist( [0 2 3], [0 0 1], [0 0 0], 3)
        >> [ (0,2), (0,1), (1,2)]
    '''
    M = M.copy()
    M[np.tril_indices(M.shape[0])] = - np.inf
    Mflat = M.flatten()
    sortIDs = np.argsort(-1 * Mflat)
    # Remove any entries that are -Inf
    sortIDs = sortIDs[Mflat[sortIDs] != -np.inf]
    if not doKeepZeros:
        # Remove any entries that are zero
        sortIDs = sortIDs[Mflat[sortIDs] != 0]
    bestrs, bestcs = np.unravel_index(sortIDs, M.shape)
    return bestrs[:nPairs].tolist(), bestcs[:nPairs].tolist()


def calcScoreMatrix_wholeELBO(curModel, SS, excludePairs=list(), M=None):
    ''' Calculate upper-tri matrix of exact ELBO gap for each candidate pair

    Returns
    ---------
    M : 2D array, size K x K. Upper triangular entries carry the content.
        M[j,k] is positive iff merging j,k improves the ELBO
                  0 otherwise
    Mraw : 2D array, size K x K. Uppert tri entries carry content.
        Mraw[j,k] gives the scalar ELBO gap for the potential merge of j,k
    '''
    K = SS.K
    if M is None:
        AGap = curModel.allocModel.calcHardMergeGap_AllPairs(SS)
        OGap = curModel.obsModel.calcHardMergeGap_AllPairs(SS)
        Mraw = AGap + OGap
        nUpdated = (SS.K * (SS.K - 1)) / 2
    else:
        assert M.shape[0] == K
        assert M.shape[1] == K
        nZeroEntry = np.sum(M == 0) - K - K * (K - 1) / 2
        assert nZeroEntry >= 0
        aList, bList = _scorematrix2rankedlist_greedy(M, SS.K + nZeroEntry,
                                                      doKeepZeros=True)
        pairList = list(zip(aList, bList))
        AGap = curModel.allocModel.calcHardMergeGap_SpecificPairs(SS, pairList)
        OGap = curModel.obsModel.calcHardMergeGap_SpecificPairs(SS, pairList)
        M[aList, bList] = AGap + OGap
        Mraw = M
        nUpdated = len(pairList)

    MergeLogger.log('MERGE ScoreMat Updates: %d entries.' % (nUpdated),
                    level='debug')

    Mraw[np.triu_indices(K, 1)] += ELBO_GAP_ACCEPT_TOL
    M = Mraw.copy()
    M[M < 0] = 0
    return M, Mraw


def calcScoreMatrix_corr(SS, MINCORR=0.05, MINVAL=1e-8):
    ''' Calculate Score matrix using correlation cues.

    Returns
    -------
    CorrMat : 2D array, size K x K
        CorrMat[j,k] = correlation coef for comps j,k
    '''
    K = SS.K
    Smat = SS.getSelectionTerm('DocTopicPairMat')
    svec = SS.getSelectionTerm('DocTopicSum')

    nanIDs = np.isnan(Smat)
    Smat[nanIDs] = 0
    svec[np.isnan(svec)] = 0
    offlimitcompIDs = np.logical_or(np.isnan(svec), svec < MINVAL)

    CovMat = Smat / SS.nDoc - np.outer(svec / SS.nDoc, svec / SS.nDoc)
    varc = np.diag(CovMat)

    sqrtc = np.sqrt(varc)
    sqrtc[offlimitcompIDs] = MINVAL

    assert sqrtc.min() >= MINVAL
    CorrMat = CovMat / np.outer(sqrtc, sqrtc)

    # Now, filter to leave only *positive* entries in upper diagonal
    #  we shouldn't even bother trying to merge topics
    #  with negative or nearly zero correlations
    CorrMat[np.tril_indices(K)] = 0
    CorrMat[CorrMat < MINCORR] = 0

    CorrMat[nanIDs] = 0

    return CorrMat


def calcScoreMatrix_corrLimitDegree(SS, MINCORR=0.05, N=3):
    ''' Score candidate merge pairs favoring correlations.

    Returns
    -------
    M : 2D array, size K x K
        M[j,k] provides score in [0, 1] for each pair of comps (j,k)
        larger score indicates better candidate for merge
    '''
    M = calcScoreMatrix_corr(SS)
    thrvec = np.linspace(MINCORR, 1.0, 10)
    fixedPairIDs = list()
    for tt in range(thrvec.size - 1, 0, -1):
        thrSm = thrvec[tt - 1]
        thrBig = thrvec[tt]
        A = np.logical_and(M > thrSm, M < thrBig)
        pairIDs = selectPairsUsingAtMostNOfEachComp(A, fixedPairIDs, N=N)
        fixedPairIDs = fixedPairIDs + pairIDs
    Mlimit = np.zeros_like(M)
    if len(fixedPairIDs) == 0:
        return Mlimit
    x, y = list(zip(*fixedPairIDs))
    Mlimit[x, y] = M[x, y]
    return Mlimit


def calcScoreMatrix_corrOrEmpty(SS, EMPTYTHR=100):
    ''' Score candidate merge pairs favoring correlations or empty components

    Returns
    -------
    M : 2D array, size K x K
        M[j,k] provides score in [0, 1] for each pair of comps (j,k)
        larger score indicates better candidate for merge
    '''
    # 1) Use correlation scores
    M = calcScoreMatrix_corr(SS)

    # 2) Add in pairs of (large mass, small mass)
    Nvec = None
    if hasattr(SS, 'N'):
        Nvec = SS.N
    elif hasattr(SS, 'SumWordCounts'):
        Nvec = SS.SumWordCounts

    assert Nvec is not None
    sortIDs = np.argsort(Nvec)
    emptyScores = np.zeros(SS.K)
    for ii in range(SS.K / 2):
        worstID = sortIDs[ii]
        bestID = sortIDs[-(ii + 1)]
        if Nvec[worstID] < EMPTYTHR and Nvec[bestID] > EMPTYTHR:
            # Want to prefer trying *larger* comps before smaller ones
            # So boost the score of larger comps slightly
            M[worstID, bestID] = 0.5 + 0.1 * Nvec[worstID] / Nvec.sum()
            M[bestID, worstID] = 0.5 + 0.1 * Nvec[worstID] / Nvec.sum()
            if Nvec[worstID] > EMPTYTHR:
                break
            emptyScores[worstID] = Nvec[worstID] / Nvec.sum()

    # 3) Add in pairs of (small mass, small mass)
    emptyIDs = np.flatnonzero(emptyScores)
    nEmpty = emptyIDs.size
    for jID in range(nEmpty - 1):
        for kID in range(jID + 1, nEmpty):
            j = emptyIDs[jID]
            k = emptyIDs[kID]
            M[j, k] = 0.4 + 0.1 * (emptyScores[j] + emptyScores[k])
    return M


def calcScoreMatrix_marglik(curModel, SS, excludePairs):
    K = SS.K
    M = np.zeros((K, K))
    excludeSet = set(excludePairs)
    myCalculator = MargLikScoreCalculator()
    for kA in range(K):
        for kB in range(kA + 1, K):
            if (kA, kB) not in excludeSet:
                M[kA, kB] = myCalculator._calcMScoreForCandidatePair(
                    curModel, SS, kA, kB)
    return M


class MargLikScoreCalculator(object):

    ''' Calculate marglik scores quickly by caching
    '''

    def __init__(self):
        self.MScores = dict()
        self.PairMScores = dict()

    def _calcMScoreForCandidatePair(self, hmodel, SS, kA, kB):
        logmA = self._calcLogMargLikForComp(hmodel, SS, kA)
        logmB = self._calcLogMargLikForComp(hmodel, SS, kB)
        logmAB = self._calcLogMargLikForPair(hmodel, SS, kA, kB)
        return logmAB - logmA - logmB

    def _calcLogMargLikForComp(self, hmodel, SS, kA):
        if kA in self.MScores:
            return self.MScores[kA]
        mA = hmodel.obsModel.calcLogMargLikForComp(
            SS, kA, doNormConstOnly=True)
        self.MScores[kA] = mA
        return mA

    def _calcLogMargLikForPair(self, hmodel, SS, kA, kB):
        if (kA, kB) in self.PairMScores:
            return self.PairMScores[(kA, kB)]
        elif (kB, kA) in self.PairMScores:
            return self.PairMScores[(kB, kA)]
        else:
            mAB = hmodel.obsModel.calcLogMargLikForComp(
                SS, kA, kB, doNormConstOnly=True)
            self.PairMScores[(kA, kB)] = mAB
            return mAB

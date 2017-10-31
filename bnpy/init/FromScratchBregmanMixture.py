from builtins import *
import numpy as np
import re
import time
import bnpy.data
from bnpy.util.OptimizerForPi import \
    estimatePiForDoc_frankwolfe, \
    estimatePiForDoc_graddescent, \
    pi2str
from .FromTruth import \
    convertLPFromHardToSoft, \
    convertLPFromTokensToDocs, \
    convertLPFromDocsToTokens, \
    convertLPFromDocsToTypes
from .FromScratchBregman import makeDataSubsetByThresholdResp

def init_global_params(hmodel, Data,
        initObsModelScale=0.0,
        **kwargs):
    ''' Initialize parameters of observation model.

    Post Condition
    --------------
    hmodel internal parameters updated to reflect sufficient statistics.
    '''
    kwargs['init_setOneToPriorMean'] = 0
    # TODO: Can we do one or more refinement iters??
    kwargs['init_NiterForBregmanKMeans'] = 0

    extrafields = kwargs['initname'].split("+")
    for key in extrafields[1:]:
        m = re.match(
            r"(?P<name>[a-zA-Z]+)(?P<value>.+)$", key)
        name = m.group('name')
        value = m.group('value')
        if name.count("lam"):
            initObsModelScale = float(value)
        elif name.count("setlasttoprior"):
            kwargs['init_setOneToPriorMean'] = int(value)
        elif name.count("distexp"):
            kwargs['init_distexp'] = float(value)
        elif name.count("iter"):
            pass
    # Obtain initial suff statistics
    SS, Info = initSS_BregmanMixture(
        Data, hmodel, includeAllocSummary=False, **kwargs)

    # Add in extra initialization mass, if needed
    if hasattr(SS, 'WordCounts') and initObsModelScale > 0:
        SS.WordCounts += initObsModelScale

    # Execute global step from these stats
    hmodel.obsModel.update_global_params(SS)
    # Finally, initialize allocation model params
    hmodel.allocModel.init_global_params(Data, **kwargs)
    Info['targetSS'] = SS
    return Info


def initSS_BregmanMixture(
        Dslice=None,
        curModel=None,
        curLPslice=None,
        K=5,
        ktarget=None,
        seed=0,
        includeAllocSummary=False,
        NiterForBregmanKMeans=0,
        setOneToPriorMean=0,
        logFunc=None,
        **kwargs):
    ''' Create observation model statistics via Breg. distance sampling.

    Returns
    -------
    xSS : SuffStatBag
    DebugInfo : dict
        contains info about provenance of this initialization.
    '''
    # Reformat any keyword argument to drop
    # prefix of 'b_' or 'init_',
    # storing the result back into the kwargs dict
    for key, val in list(kwargs.items()):
        if key.startswith('b_'):
            newkey = key[2:]
            kwargs[newkey] = val
            del kwargs[key]
        elif key.startswith('init_'):
            newkey = key[5:]
            kwargs[newkey] = val
            del kwargs[key]
    if 'setOneToPriorMean' in kwargs:
        setOneToPriorMean = kwargs['setOneToPriorMean']
    else:
        kwargs['setOneToPriorMean'] = setOneToPriorMean
    Niter = np.maximum(NiterForBregmanKMeans, 0)

    if logFunc:
        logFunc("Preparing target dataset for Bregman mixture analysis...")
    DebugInfo, targetData, targetX, targetW, chosenRespIDs = \
        makeDataSubsetByThresholdResp(
            Dslice,
            curModel,
            curLPslice,
            ktarget,
            K=K,
            **kwargs)
    if logFunc:
        logFunc(DebugInfo['targetAssemblyMsg'])
    if targetData is None:
        assert 'errorMsg' in DebugInfo
        return None, DebugInfo
    K = np.minimum(K, targetX.shape[0])
    if logFunc:
        msg = "Running Bregman mixture with K=%d for %d iters" % (
            K, Niter)
        if setOneToPriorMean:
            msg += ", with initial prior mean cluster"
        logFunc(msg)

    # Perform plusplus initialization + Kmeans clustering
    targetZ, Mu, minDiv, DivDataVec, Lscores = initKMeans_BregmanMixture(
        targetData, K, curModel.obsModel,
        seed=seed,
        **kwargs)
    # Convert labels in Z to compactly use all ints from 0, 1, ... Kused
    # Then translate these into a proper 'resp' 2D array,
    # where resp[n,k] = w[k] if z[n] = k, and 0 otherwise
    xtargetLP, _ = convertLPFromHardToSoft(
        dict(Z=targetZ), targetData, initGarbageState=0, returnZ=1)
    print(targetZ, '<<<')
    if K == 1:
        xtargetLP['resp'] = np.zeros((xtargetLP['resp'].shape[0], 1))

    if isinstance(Dslice, bnpy.data.BagOfWordsData):
        if curModel.obsModel.DataAtomType.count('word'):
            if curModel.getObsModelName().count('Bern'):
                xtargetLP = convertLPFromDocsToTypes(xtargetLP, targetData)
            else:
                xtargetLP = convertLPFromDocsToTokens(xtargetLP, targetData)
    # Summarize the local parameters
    if includeAllocSummary and Niter > 0:
        if hasattr(curModel.allocModel, 'initLPFromResp'):
            xtargetLP = curModel.allocModel.initLPFromResp(
                targetData, xtargetLP)
        xSS = curModel.get_global_suff_stats(
            targetData, xtargetLP)
    else:
        xSS = curModel.obsModel.get_global_suff_stats(
            targetData, None, xtargetLP)

    if setOneToPriorMean:
        assert len(Mu) == xSS.K - 1
        # First cluster Mu0 needs to be initialized to prior mean
        xSS.insertEmptyComps(1)
        xSS.reorderComps(
            np.hstack([xSS.K, np.arange(xSS.K)]))
    else:
        assert len(Mu) == xSS.K
        assert np.allclose(
            np.unique(targetZ[targetZ >= 0]),
            np.arange(xSS.K))

    # Reorder the components from big to small
    oldids_bigtosmall = np.argsort(-1 * xSS.getCountVec())
    xSS.reorderComps(oldids_bigtosmall)
    # Be sure to account for the sorting that just happened.
    # By fixing up the cluster means Mu and assignments Z
    Mu = [Mu[k] for k in oldids_bigtosmall]
    neworder = np.arange(xSS.K)
    print(neworder)
    print(oldids_bigtosmall)
    old2newID=dict(list(zip(oldids_bigtosmall, neworder)))
    targetZnew = -1 * np.ones_like(targetZ)
    for oldk in range(xSS.K):
        old_mask = targetZ == oldk
        targetZnew[old_mask] = old2newID[oldk]
    assert np.allclose(len(Mu), xSS.K)
    if logFunc:
        logFunc('Bregman k-means DONE. Delivered %d non-empty clusters' % (
            xSS.K))
    # Package up algorithm final state and Lscore trace
    DebugInfo.update(dict(
        minDiv=minDiv,
        targetZ=targetZnew,
        targetData=targetData,
        Mu=Mu,
        Lscores=Lscores))
    return xSS, DebugInfo

def initKMeans_BregmanMixture(Data, K, obsModel, seed=0,
        setOneToPriorMean=0,
        distexp=1.0,
        alpha=1.001,
        verbose=True,
        optim_method='frankwolfe',
        **kwargs):
    '''

    Returns
    -------
    Z : 1D array
    '''
    starttime = time.time()
    PRNG = np.random.RandomState(int(seed))
    X = Data.getDocTypeCountMatrix()
    V = Data.vocab_size
    # Select first cluster mean
    chosenZ = np.zeros(K, dtype=np.int32)
    if setOneToPriorMean:
        chosenZ[0] = -1
        emptyXvec = np.zeros_like(X[0])
        Mu0 = obsModel.calcSmoothedMu(emptyXvec)
    else:
        chosenZ[0] = PRNG.choice(X.shape[0])
        Mu0 = obsModel.calcSmoothedMu(X[chosenZ[0]])
    # Initialize list to hold all Mu values
    Mu = [None for k in range(K)]
    Mu[0] = Mu0
    # Compute minDiv
    minDiv, DivDataVec = obsModel.calcSmoothedBregDiv(
        X=X, Mu=Mu0,
        returnDivDataVec=True,
        return1D=True,
        smoothFrac=1.0)
    Pi = np.ones((X.shape[0], K))
    scoreVsK = list()
    for k in range(1, K):
        # Do not select any doc more than once
        minDiv[chosenZ[setOneToPriorMean:k]] = 0
        # Total up the score
        sum_minDiv = np.sum(minDiv)
        scoreVsK.append(sum_minDiv)
        if sum_minDiv == 0.0:
            # Duplicate rows corner case
            # Some rows of X may be exact copies,
            # leading to all minDiv being zero if chosen covers all copies
            chosenZ = chosenZ[:k]
            for emptyk in reversed(list(range(k, K))):
                # Remove remaining entries in the Mu list,
                # so its total size is now k, not K
                Mu.pop(emptyk)
            assert len(Mu) == chosenZ.size
            break
        elif sum_minDiv < 0 or not np.isfinite(sum_minDiv):
            raise ValueError("sum_minDiv not valid: %f" % (sum_minDiv))
        if distexp >= 9:
            chosenZ[k] = np.argmax(minDiv)
        else:
            if distexp > 1:
                minDiv = minDiv**distexp
                sum_minDiv = np.sum(minDiv)
            pvec = minDiv / sum_minDiv
            chosenZ[k] = PRNG.choice(X.shape[0], p=pvec)
        Mu[k] = obsModel.calcSmoothedMu(X[chosenZ[k]])

        # Compute next value of pi
        Pi, minDiv = estimatePiAndDiv_ManyDocs(
            Data, obsModel, Mu,
            k=k+1,
            Pi=Pi,
            alpha=alpha,
            smoothVec='lam',
            minDiv=minDiv,
            DivDataVec=DivDataVec,
            optim_method=optim_method)
        time_k = time.time()
        if verbose:
            print(" completed round %3d/%d after %6.1f sec" % (
                k+1, K, time_k - starttime))
    # Every selected doc should have zero distance
    minDiv[chosenZ[setOneToPriorMean:]] = 0
    # Compute final score and add to the list
    scoreVsK.append(np.sum(minDiv))
    # Generally, we'd expect scores to be monotonically decreasing
    # However, we optimize pi_d under a nonconvex objective
    # *and* this optimization does not (yet) use smoothing
    # So there may be *slight* hiccups
    #assert np.all(np.diff(scoreVsK) >= -1e-6)

    Z = -1 * np.ones(Data.nDoc)
    if setOneToPriorMean:
        Z[chosenZ[1:]] = np.arange(chosenZ.size - 1)
    else:
        Z[chosenZ] = np.arange(chosenZ.size)
    # Without full pass through dataset, many items not assigned
    # which we indicated with Z value of -1
    # Should ignore this when counting states
    uniqueZ = np.unique(Z)
    uniqueZ = uniqueZ[uniqueZ >= 0]
    if setOneToPriorMean:
        assert len(Mu) == uniqueZ.size + 1
    else:
        assert len(Mu) == uniqueZ.size

    return Z, Mu, minDiv, np.sum(DivDataVec), scoreVsK


def estimatePiAndDiv_ManyDocs(Data, obsModel, Mu,
        Pi=None,
        k=None,
        alpha=1.0,
        optim_method='frankwolfe',
        doActiveOnly=True,
        DivDataVec=None,
        smoothVec='lam',
        maxiter=100,
        minDiv=None):
    ''' Estimate doc-topic probs for many docs, with corresponding divergence

    Returns
    -------
    Pi : 2D array, size D x K
    minDiv : 1D array, size D
        minDiv[d] : divergence from closest convex combination of topics in Mu
    '''
    K = len(Mu)
    if k is None:
        k = K
    if isinstance(Mu, list):
        topics = np.vstack(Mu[:k])
    else:
        topics = Mu[:k]

    if Pi is None:
        Pi = np.ones((Data.nDoc, K))
    if minDiv is None:
        minDiv = np.zeros(Data.nDoc)
    for d in range(Data.nDoc):
        start_d = Data.doc_range[d]
        stop_d = Data.doc_range[d+1]
        wids_d = Data.word_id[start_d:stop_d]
        wcts_d = Data.word_count[start_d:stop_d]

        if doActiveOnly:
            activeIDs_d = np.flatnonzero(Pi[d, :k] > .01)
            if activeIDs_d[-1] != k-1:
                activeIDs_d = np.append(activeIDs_d, k-1)
        else:
            activeIDs_d = np.arange(k)
        assert activeIDs_d.size >= 1
        assert activeIDs_d.size <= k

        topics_d = topics[activeIDs_d,:]
        assert topics_d.shape[0] <= k

        initpiVec_d = Pi[d, activeIDs_d].copy()
        initpiVec_d[-1] = 0.1
        initpiVec_d[:-1] *= 0.9
        initpiVec_d /= initpiVec_d.sum()
        assert np.allclose(initpiVec_d.sum(), 1.0)

        if optim_method == 'frankwolfe':
            piVec_d = estimatePiForDoc_frankwolfe(
                ids_U=wids_d,
                cts_U=wcts_d,
                topics_KV=topics_d,
                initpiVec_K=initpiVec_d,
                alpha=alpha,
                seed=(k*101 + d),
                maxiter=maxiter,
                returnFuncValAndInfo=False,
                verbose=False)
            piVec_d *= Pi[d, activeIDs_d[:-1]].sum()
            Pi[d, activeIDs_d] = piVec_d
        else:
            Pi[d, :k], _, _ = estimatePiForDoc_graddescent(
                ids_d=wids_d,
                cts_d=wcts_d,
                topics=topics,
                alpha=alpha,
                scale=1.0,
                piInit=None)

        assert np.allclose(Pi[d,:k].sum(), 1.0)
        minDiv[d] = -1 * np.inner(wcts_d,
            np.log(np.dot(Pi[d,:k], topics[:, wids_d])))

    minDiv_check = -1 * np.sum(
        Data.getDocTypeCountMatrix() *
        np.log(np.dot(Pi[:, :k], topics)), axis=1)
    assert np.allclose(minDiv, minDiv_check)

    if isinstance(smoothVec, str) and smoothVec.count('lam'):
        minDiv -= np.dot(np.log(np.dot(Pi[:, :k], topics)), obsModel.Prior.lam)
    elif isinstance(smoothVec, np.ndarray):
        minDiv -= np.dot(np.log(np.dot(Pi[:, :k], topics)), smoothVec)
    if DivDataVec is not None:
        minDiv += DivDataVec
    assert np.min(minDiv) > -1e-6
    np.maximum(minDiv, 0, out=minDiv)
    return Pi, minDiv

if __name__ == '__main__':
    import CleanBarsK10
    Data = CleanBarsK10.get_data(nDocTotal=100, nWordsPerDoc=500)
    K = 3

    hmodel, Info = bnpy.run(Data, 'DPMixtureModel', 'Mult', 'memoVB',
        initname='bregmankmeans+iter0',
        K=K,
        nLap=0)

    obsModel = hmodel.obsModel.copy()
    bestMu = None
    bestScore = np.inf
    nTrial = 1
    for trial in range(nTrial):
        chosenZ, Mu, minDiv, sumDataTerm, scoreVsK = initKMeans_BregmanMixture(
            Data, K, obsModel, seed=trial)
        score = np.sum(minDiv)
        print("init %d/%d : sum(minDiv) %8.2f" % (
            trial+1, nTrial, np.sum(minDiv)))
        if score < bestScore:
            bestScore = score
            bestMu = Mu
            print("*** New best")

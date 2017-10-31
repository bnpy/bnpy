'''
FromScratchBregman.py

Initialize suff stats for observation models via Bregman clustering.
'''
from builtins import *
import re
import numpy as np
import bnpy.data

from bnpy.util import split_str_into_fixed_width_lines
from .FromTruth import \
    convertLPFromHardToSoft, \
    convertLPFromTokensToDocs, \
    convertLPFromDocsToTokens, \
    convertLPFromDocsToTypes

def init_global_params(hmodel, Data, **kwargs):
    ''' Initialize parameters of observation model.

    Post Condition
    --------------
    hmodel internal parameters updated to reflect sufficient statistics.
    '''
    if 'initObsModelScale' in kwargs:
        initObsModelScale = kwargs['initObsModelScale']
    else:
        initObsModelScale = 0
    #if kwargs['initname'].lower().count('priormean'):
    #    kwargs['init_setOneToPriorMean'] = 1

    extrafields = kwargs['initname'].split("+")
    for key in extrafields[1:]:
        m = re.match(
            r"(?P<name>[a-zA-Z]+)(?P<value>.+)$", key)
        name = m.group('name')
        value = m.group('value')
        if name.count("lam"):
            initObsModelScale = float(value)
        elif name.count("distexp"):
            kwargs['init_distexp'] = float(value)
        elif name.count("setlasttoprior"):
            kwargs['init_setOneToPriorMean'] = int(value)
        elif name.count("iter"):
            kwargs['init_NiterForBregmanKMeans'] = int(value)

            if 'logFunc' not in kwargs:
                def logFunc(msg):
                    print(msg)
                kwargs['logFunc'] = logFunc

    '''
    if kwargs['initname'].count('+'):
        kwargs['init_NiterForBregmanKMeans'] = \
            int(kwargs['initname'].split('+')[1])
        if 'logFunc' not in kwargs:
            def logFunc(msg):
                print msg
            kwargs['logFunc'] = logFunc
    '''
    # Determine initial SS
    SS, Info = initSS_BregmanDiv(
        Data, hmodel, includeAllocSummary=True, **kwargs)
    # Add in extra initialization mass, if needed
    if hasattr(SS, 'WordCounts') and initObsModelScale > 0:
        SS.WordCounts += initObsModelScale
    # Execute global step on obsModel from these stats
    hmodel.obsModel.update_global_params(SS)
    # Finally, initialize allocation model params
    if kwargs['init_NiterForBregmanKMeans'] > 0:
        hmodel.allocModel.update_global_params(SS)
    else:
        hmodel.allocModel.init_global_params(Data, **kwargs)
    Info['targetSS'] = SS
    return Info

def initSS_BregmanDiv(
        Dslice=None,
        curModel=None,
        curLPslice=None,
        K=5,
        ktarget=None,
        seed=0,
        includeAllocSummary=False,
        NiterForBregmanKMeans=1,
        logFunc=None,
        **kwargs):
    ''' Create observation model statistics via Breg. distance sampling.

    Args
    ------
    Data : bnpy dataset
        dataset
    curModel : bnpy HModel
        must at least have defined obsModel prior distribution
    curLPslice : None or LP dict
        if None, will use entire dataset

    ktarget : int
        id of specific cluster to target within curModel
        if None, will use entire dataset

    Keyword args
    ------------
    TODO

    Returns
    -------
    xSS : SuffStatBag
    DebugInfo : dict
        contains info about provenance of this initialization.
    '''
    # Reformat any keyword argument to drop
    # prefix of 'b_' or 'init_'
    for key, val in list(kwargs.items()):
        if key.startswith('b_'):
            newkey = key[2:]
            kwargs[newkey] = val
            del kwargs[key]
        elif key.startswith('init_'):
            newkey = key[5:]
            kwargs[newkey] = val
            del kwargs[key]
    if 'NiterForBregmanKMeans' in kwargs:
        NiterForBregmanKMeans = kwargs['NiterForBregmanKMeans']
    Niter = np.maximum(NiterForBregmanKMeans, 0)

    if logFunc:
        logFunc("Preparing target dataset for Bregman k-means analysis...")
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
        msg = "Running Bregman k-means with K=%d for %d iters" % (
            K, Niter)
        if 'setOneToPriorMean' in kwargs and kwargs['setOneToPriorMean']:
            msg += ", with initial prior mean cluster"
        logFunc(msg)

    # Perform plusplus initialization + Kmeans clustering
    targetZ, Mu, Lscores = runKMeans_BregmanDiv(
        targetX, K, curModel.obsModel,
        W=targetW,
        Niter=Niter,
        logFunc=logFunc,
        seed=seed,
        **kwargs)
    # Convert labels in Z to compactly use all ints from 0, 1, ... Kused
    # Then translate these into a proper 'resp' 2D array,
    # where resp[n,k] = w[k] if z[n] = k, and 0 otherwise
    xtargetLP, targetZ = convertLPFromHardToSoft(
        dict(Z=targetZ), targetData, initGarbageState=0, returnZ=1)
    if isinstance(Dslice, bnpy.data.BagOfWordsData):
        if curModel.obsModel.DataAtomType.count('word'):
            if curModel.getObsModelName().count('Bern'):
                xtargetLP = convertLPFromDocsToTypes(xtargetLP, targetData)
            else:
                xtargetLP = convertLPFromDocsToTokens(xtargetLP, targetData)
    if curLPslice is not None:
        if 'resp' in curLPslice:
            xtargetLP['resp'] *= \
                curLPslice['resp'][chosenRespIDs, ktarget][:,np.newaxis]
            # Verify that initial xLP resp is a subset of curLP's resp,
            # leaving out only the docs that didnt have enough tokens.
            assert np.all(xtargetLP['resp'].sum(axis=1) <= \
                curLPslice['resp'][chosenRespIDs, ktarget] + 1e-5)

        elif 'spR' in curLPslice:
            inds = chosenRespIDs * curLPslice['nnzPerRow']
            xtargetLP['resp'] *= curLPslice['spR'].data[inds][:,np.newaxis]

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
        if 'setOneToPriorMean' in kwargs and kwargs['setOneToPriorMean']:
            neworder = np.hstack([xSS.K, np.arange(xSS.K)])
            xSS.insertEmptyComps(1)
            xSS.reorderComps(neworder)
        else:
            assert np.allclose(np.unique(targetZ), np.arange(xSS.K))
        assert np.allclose(len(Mu), xSS.K)
    # Reorder the components from big to small
    oldids_bigtosmall = np.argsort(-1 * xSS.getCountVec())
    xSS.reorderComps(oldids_bigtosmall)
    # Be sure to account for the sorting that just happened.
    # By fixing up the cluster means Mu and assignments Z
    Mu = [Mu[k] for k in oldids_bigtosmall]
    neworder = np.arange(xSS.K)
    old2newID=dict(list(zip(oldids_bigtosmall, neworder)))
    targetZnew = -1 * np.ones_like(targetZ)
    for oldk in range(xSS.K):
        old_mask = targetZ == oldk
        targetZnew[old_mask] = old2newID[oldk]
    assert np.all(targetZnew >= 0)
    assert np.allclose(len(Mu), xSS.K)
    if logFunc:
        logFunc('Bregman k-means DONE. Delivered %d non-empty clusters' % (
            xSS.K))
    # Package up algorithm final state and Lscore trace
    DebugInfo.update(dict(
        targetZ=targetZnew,
        targetData=targetData,
        Mu=Mu,
        Lscores=Lscores))
    return xSS, DebugInfo

def runKMeans_BregmanDiv(
        X, K, obsModel, W=None,
        Niter=100, seed=0, init='plusplus',
        smoothFracInit=1.0, smoothFrac=0,
        logFunc=None, eps=1e-10,
        setOneToPriorMean=0,
        distexp=1.0,
        assert_monotonic=True,
        **kwargs):
    ''' Run hard clustering algorithm to find K clusters.

    Returns
    -------
    Z : 1D array, size N
    Mu : 2D array, size K x D
    Lscores : 1D array, size Niter
    '''
    chosenZ, Mu, _, _ = initKMeans_BregmanDiv(
        X, K, obsModel, W=W, seed=seed,
        smoothFrac=smoothFracInit,
        distexp=distexp,
        setOneToPriorMean=setOneToPriorMean)
    # Make sure we update K to reflect the returned value.
    # initKMeans_BregmanDiv will return fewer than K clusters
    # in some edge cases, like when data matrix X has duplicate rows
    # and specified K is larger than the number of unique rows.
    K = len(Mu)
    assert K > 0
    assert Niter >= 0
    if Niter == 0:
        Z = -1 * np.ones(X.shape[0])
        if chosenZ[0] == -1:
            Z[chosenZ[1:]] = np.arange(chosenZ.size - 1)
        else:
            Z[chosenZ] = np.arange(chosenZ.size)
    Lscores = list()
    prevN = np.zeros(K)
    for riter in range(Niter):
        Div = obsModel.calcSmoothedBregDiv(
            X=X, Mu=Mu, W=W,
            includeOnlyFastTerms=True,
            smoothFrac=smoothFrac, eps=eps)
        Z = np.argmin(Div, axis=1)
        Ldata = Div.min(axis=1).sum()
        Lprior = obsModel.calcBregDivFromPrior(
            Mu=Mu, smoothFrac=smoothFrac).sum()
        Lscore = Ldata + Lprior
        Lscores.append(Lscore)
        # Verify objective is monotonically increasing
        if assert_monotonic:
            try:
                # Test allows small positive increases that are
                # numerically indistinguishable from zero. Don't care about these.
                assert np.all(np.diff(Lscores) <= 1e-5)
            except AssertionError:
                msg = "iter %d: Lscore %.3e" % (riter, Lscore)
                msg += '\nIn the kmeans update loop of FromScratchBregman.py'
                msg += '\nLscores not monotonically decreasing...'
                if logFunc:
                    logFunc(msg)
                else:
                    print(msg)
                assert np.all(np.diff(Lscores) <= 1e-5)

        N = np.zeros(K)
        for k in range(K):
            if W is None:
                W_k = None
                N[k] = np.sum(Z==k)
            else:
                W_k = W[Z==k]
                N[k] = np.sum(W_k)
            if N[k] > 0:
                Mu[k] = obsModel.calcSmoothedMu(X[Z==k], W_k)
            else:
                Mu[k] = obsModel.calcSmoothedMu(X=None)
        if logFunc:
            logFunc("iter %d: Lscore %.3e" % (riter, Lscore))
            if W is None:
                 str_sum_w = ' '.join(['%7.0f' % (x) for x in N])
            else:
                 assert np.allclose(N.sum(), W.sum())
                 str_sum_w = ' '.join(['%7.2f' % (x) for x in N])
            str_sum_w = split_str_into_fixed_width_lines(str_sum_w, tostr=True)
            logFunc(str_sum_w)
        if np.max(np.abs(N - prevN)) == 0:
            break
        prevN[:] = N

    uniqueZ = np.unique(Z)
    if Niter > 0:
        # In case a cluster was pushed to zero
        if uniqueZ.size < len(Mu):
            Mu = [Mu[k] for k in uniqueZ]
    else:
        # Without full pass through dataset, many items not assigned
        # which we indicated with Z value of -1
        # Should ignore this when counting states
        uniqueZ = uniqueZ[uniqueZ >= 0]
    assert len(Mu) == uniqueZ.size
    return Z, Mu, np.asarray(Lscores)

def initKMeans_BregmanDiv(
        X, K, obsModel, W=None, seed=0, smoothFrac=1.0,
        distexp=1.0,
        setOneToPriorMean=0):
    ''' Initialize cluster means Mu for K clusters.

    Returns
    -------
    chosenZ : 1D array, size K
        int ids of atoms selected
    Mu : list of size K
        each entry is a tuple of ND arrays
    minDiv : 1D array, size N
    '''
    PRNG = np.random.RandomState(int(seed))
    N = X.shape[0]
    if W is None:
        W = np.ones(N)
    chosenZ = np.zeros(K, dtype=np.int32)
    if setOneToPriorMean:
        chosenZ[0] = -1
        emptyXvec = np.zeros_like(X[0])
        Mu0 = obsModel.calcSmoothedMu(emptyXvec)
    else:
        chosenZ[0] = PRNG.choice(N, p=W/np.sum(W))
        Mu0 = obsModel.calcSmoothedMu(X[chosenZ[0]], W=W[chosenZ[0]])

    # Initialize list to hold all Mu values
    Mu = [None for k in range(K)]
    Mu[0] = Mu0
    # Compute minDiv
    minDiv, DivDataVec = obsModel.calcSmoothedBregDiv(
        X=X, Mu=Mu0, W=W,
        returnDivDataVec=True,
        return1D=True,
        smoothFrac=smoothFrac)
    if not setOneToPriorMean:
        minDiv[chosenZ[0]] = 0

    # Sample each cluster id using distance heuristic
    for k in range(1, K):
        sum_minDiv = np.sum(minDiv)
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
            chosenZ[k] = PRNG.choice(N, p=pvec)

        Mu[k] = obsModel.calcSmoothedMu(X[chosenZ[k]], W=W[chosenZ[k]])
        curDiv = obsModel.calcSmoothedBregDiv(
            X=X, Mu=Mu[k], W=W,
            DivDataVec=DivDataVec,
            return1D=True,
            smoothFrac=smoothFrac)
        curDiv[chosenZ[k]] = 0
        minDiv = np.minimum(minDiv, curDiv)
    return chosenZ, Mu, minDiv, np.sum(DivDataVec)




def makeDataSubsetByThresholdResp(
        Data, curModel,
        curLP=None,
        ktarget=None,
        K=None,
        minRespForEachTargetAtom=None,
        **kwargs):
    ''' Make subset of provided dataset by thresholding assignments.

    Args
    ----
    Data : bnpy dataset
    curLP : dict of local parameters
    ktarget : integer id of cluster to target, in {0, 1, ... K-1}

    Returns
    -------
    DebugInfo : dict
    targetData : bnpy data object, representing data subset
    targetX : 2D array, size N x K, whose rows will be clustered
    targetW : 1D array, size N
        None indicates uniform weight on all data items
    chosenRespIDs : 1D array, size curLP['resp'].shape[0]
        None indicates no curLP provided.
    '''
    if isinstance(Data, bnpy.data.BagOfWordsData):
        return makeDataSubsetByThresholdResp_BagOfWordsData(Data, curModel,
            curLP=curLP,
            ktarget=ktarget,
            K=K,
            minRespForEachTargetAtom=minRespForEachTargetAtom,
            **kwargs)
    assert isinstance(Data, bnpy.data.XData) or \
        isinstance(Data, bnpy.data.GroupXData)
    Natoms_total = Data.X.shape[0]
    atomType = 'atoms'
    if curLP is None:
        targetData = Data
        targetX = Data.X
        targetW = None
        chosenRespIDs = None
        Natoms_target = targetX.shape[0]
        Natoms_targetAboveThr = targetX.shape[0]
        targetAssemblyMsg = \
            "  Using all %d/%d atoms for initialization." % (
                Natoms_target, Natoms_total)
    else:
        if 'resp' in curLP:
            chosenRespIDs = np.flatnonzero(
                curLP['resp'][:,ktarget] > minRespForEachTargetAtom)
            Natoms_target = curLP['resp'][:,ktarget].sum()
        else:
            assert 'spR' in curLP
            indsThatUseTarget = np.flatnonzero(
                curLP['spR'].indices == ktarget)
            subsetAboveThr = curLP['spR'].data[indsThatUseTarget] > minRespForEachTargetAtom
            indsThatUseTarget = indsThatUseTarget[subsetAboveThr]
            chosenRespIDs = indsThatUseTarget // curLP['nnzPerRow']
            Natoms_target = curLP['spR'].data[indsThatUseTarget].sum()

        Natoms_targetAboveThr = chosenRespIDs.size
        targetAssemblyMsg = \
            "  Targeted comp has %.2f %s assigned out of %d." % (
                Natoms_target, atomType, Natoms_total) \
            + "\n  Filtering to find atoms with resp > %.2f" % (
                minRespForEachTargetAtom) \
            + "\n  Found %d atoms meeting this requirement." % (
                Natoms_targetAboveThr)

        # Raise error if target dataset not big enough.
        Keff = np.minimum(K, chosenRespIDs.size)
        if Keff <= 1 and K > 1:
            DebugInfo = dict(
                errorMsg="Filtered dataset too small." + \
                    "Wanted %d items, found %d." % (K, Keff),
                targetAssemblyMsg=targetAssemblyMsg,
                atomType=atomType,
                Natoms_total=Natoms_total,
                Natoms_target=Natoms_target,
                Natoms_targetAboveThr=Natoms_targetAboveThr,
                )
            return DebugInfo, None, None, None, None
        targetData = Data.make_subset(example_id_list=chosenRespIDs)
        targetX = targetData.X
        if curLP is None:
            targetW = None
        elif 'resp' in curLP:
            targetW = curLP['resp'][chosenRespIDs,ktarget]
        elif 'spR' in curLP:
            targetW = curLP['spR'].data[indsThatUseTarget]
        else:
            raise ValueError("if curLP specified, must have resp or spR")
    chosenDataIDs = chosenRespIDs
    DebugInfo = dict(
        targetW=targetW,
        chosenDataIDs=chosenDataIDs,
        chosenRespIDs=chosenRespIDs,
        targetAssemblyMsg=targetAssemblyMsg,
        atomType=atomType,
        Natoms_total=Natoms_total,
        Natoms_target=Natoms_target,
        Natoms_targetAboveThr=Natoms_targetAboveThr,
        )
    return DebugInfo, targetData, targetX, targetW, chosenRespIDs


def makeDataSubsetByThresholdResp_BagOfWordsData(
        Data, curModel,
        curLP=None,
        ktarget=None,
        minNumAtomsInEachTargetDoc=0,
        minRespForEachTargetAtom=0.1,
        K=0,
        **kwargs):
    ''' Make subset of provided dataset by thresholding assignments.

    Args
    ----
    Data : bnpy dataset
    curLP : dict of local parameters
    ktarget : integer id of cluster to target, in {0, 1, ... K-1}

    Returns
    -------
    DebugInfo : dict
    targetData : bnpy data object, representing data subset
    targetX : 2D array, size N x K, whose rows will be clustered
    targetW : 1D array, size N
        None indicates uniform weight on all data items
    chosenRespIDs : 1D array, size curLP['resp'].shape[0]
        None indicates no curLP provided.
    '''
    obsModelName = curModel.getObsModelName()
    if curLP is None:
        # TODO Filter docs by required minimum size
        targetAssemblyMsg = \
            "  Using entire provided dataset of %.2f docs (BagOfWordsData fmt)." % (
                Data.nDoc) \
            + "\n  No LP provided for specialized targeting."
        Natoms_targetAboveThr = Data.nDoc
        DebugInfo = dict(
            targetAssemblyMsg=targetAssemblyMsg,
            atomType='doc',
            nDoc=Data.nDoc,
            dataType='BagOfWordsData',
            obsModelName=obsModelName,
            docIDs=list(range(Data.nDoc)),
            )
        # Raise error if target dataset not big enough.
        Keff = np.minimum(K, Data.nDoc)
        if Keff <= 1 and K > 1:
            DebugInfo['errorMsg']= \
                "Filtered dataset too small." + \
                "Wanted 2 or more docs, found %d." % (K, Keff)
            return DebugInfo, None, None, None, None
        if obsModelName.count('Bern'):
            X = Data.getDocTypeBinaryMatrix()
        elif obsModelName.count('Mult'):
            X = Data.getDocTypeCountMatrix()
        else:
            raise ValueError("Unrecognized obsmodel: " + obsModelName)
        return DebugInfo, Data, X, None, None
    # Compute weights for the targeted comp
    # Need to handle special cases for clustering words and clustering docs
    targetRespVec = curLP['resp'][:,ktarget]
    if targetRespVec.size == Data.nUniqueToken:
        # HDP case : clustering individual present words
        assert 'DocTopicCount' in curLP
        targetData, docIDs, chosenRespIDs = \
            Data.makeSubsetByThresholdingWeights(
                atomWeightVec=targetRespVec,
                thr=minRespForEachTargetAtom)
        targetW = None
    elif targetRespVec.size == Data.nDoc:
        # DP case : clustering entire document-count vectors
        docIDs = np.flatnonzero(targetRespVec >= minRespForEachTargetAtom)
        chosenRespIDs = docIDs
        targetData = Data.make_subset(docMask=docIDs)
        targetW = targetRespVec[docIDs]
    elif targetRespVec.size == Data.nDoc * Data.vocab_size:
        # HDP bernoulli case : clustering all word types in every doc
        assert 'DocTopicCount' in curLP
        weightVec = -1 * np.ones(Data.nUniqueToken)
        posRespIDs = np.zeros(Data.nUniqueToken)
        for d in range(Data.nDoc):
            start = Data.doc_range[d]
            stop = Data.doc_range[d+1]
            pos_words = Data.word_id[start:stop]
            weightVec[start:stop] = targetRespVec[
                d * Data.vocab_size + pos_words]
        assert weightVec.min() >= 0.0
        targetData, docIDs, chosenPosIDs = \
            Data.makeSubsetByThresholdingWeights(
                atomWeightVec=weightVec,
                thr=minRespForEachTargetAtom)
        chosenRespIDs = list()
        for d in docIDs:
            chosenRespIDs.extend(
                list(range(d * Data.vocab_size, (d+1) * Data.vocab_size)))
        targetW = None
    else:
        raise ValueError("Should never happen")
    targetAssemblyMsg = \
        " Provided dataset of %.2f docs in BagOfWordsData format." % (
            Data.nDoc) + \
        "\n Targeting cluster at idx %d in provided LP with K=%d clusters." % (
            ktarget, curLP['resp'].shape[1]) + \
        "\n Target cluster has %.2f docs assigned with mass >%.3f." % (
            len(docIDs), minRespForEachTargetAtom)
    targetAssemblyMsg += \
        "\n " + makeSummaryStrForTargetResp(curLP['resp'][:,ktarget],
            nDoc=Data.nDoc,
            vocab_size=Data.vocab_size,
            doc_range=Data.doc_range,
            word_id=Data.word_id,
            word_count=Data.word_count,
            docIDs=docIDs,
            minRespForEachTargetAtom=minRespForEachTargetAtom)
    DebugInfo = dict(
        targetAssemblyMsg=targetAssemblyMsg,
        atomType='doc',
        dataType='BagOfWordsData',
        obsModelName=obsModelName,
        chosenRespIDs=chosenRespIDs,
        chosenDataIDs=docIDs,
        )
    # Raise error if target dataset not big enough.
    Keff = np.minimum(K, len(docIDs))
    if Keff <= 1 and K > 1:
        DebugInfo['errorMsg'] = "Dataset too small to cluster." + \
            " Wanted 2 or more docs, found %d." % (Keff)
        return DebugInfo, None, None, None, None
    # Make nDoc x vocab_size array
    if obsModelName.count('Mult'):
        targetX = targetData.getDocTypeCountMatrix()
    else:
        targetX = targetData.getDocTypeBinaryMatrix()
    emptyRows = np.flatnonzero(targetX.sum(axis=1) < 1.0)
    if emptyRows.size > 0:
        raise ValueError('WHOA! Found some empty rows in the targetX')
    return DebugInfo, targetData, targetX, targetW, chosenRespIDs

def makeSummaryStrForTargetResp(respVec,
        nDoc=None,
        vocab_size=None,
        doc_range=None,
        word_id=None,
        word_count=None,
        docIDs=None,
        minRespForEachTargetAtom=None):
    ''' Make human-readable description of targeted resp values.

    Args
    ----
    respVec :
    doc_range : 1D array
        Attrib of ORIGINAL dataset.
    word_count : 1D array
        Attrib of ORIGINAL dataset.

    Returns
    -------
    s : str
    '''
    if word_count is not None and \
            respVec.size == doc_range[-1] and respVec.size == word_count.size:
        # BagOfWordsData with Mult likelihood and HDP clustering objective
        msg = "Target docs contain subset of words in corrsp. original doc."
        nAboveList = list()
        massAboveList = list()
        for d in docIDs:
            start = doc_range[d]
            stop = doc_range[d+1]
            mass = np.sum(respVec[start:stop], axis=0)
            aboveMask = respVec[start:stop] >= minRespForEachTargetAtom
            massAbove = np.sum(
                respVec[start:stop][aboveMask] * \
                word_count[start:stop][aboveMask], axis=0)
            nAboveList.append(np.sum(aboveMask))
            massAboveList.append(massAbove)

        msg += "\n"
        msg += " Across target docs, distrib. of nDistinctTypes  :"
        for p in [0, 10, 50, 90, 100]:
            md = np.percentile(nAboveList, p)
            msg += " %3d%% %5d  " % (p, md)
        msg += "\n"
        msg += "                              of mass (wc * resp):"
        for p in [0, 10, 50, 90, 100]:
            md = np.percentile(massAboveList, p)
            msg += " %3d%% %7.1f" % (p, md)
    elif word_count is not None and respVec.size == nDoc:
        # BagOfWordsData with Mult likelihood and DP clustering objective
        msg = "Target docs are strictly equal to corresp. original docs."
        massList = list()
        for d in docIDs:
            massList.append(respVec[d])
        msg += "\n Total resp mass on target docs: %.3f" % (np.sum(massList))
        msg += "\n Across target docs, distrib of respMass  :"
        for p in [0, 10, 50, 90, 100]:
            md = np.percentile(massList, p)
            msg += " %3d%% %4.3f" % (p, md)
    elif word_count is not None and respVec.size == nDoc * vocab_size:
        msg = "Target docs hold subset of BINARY words in corresp. orig. doc."
        nAboveList = list()
        massAboveList = list()
        totalMassTtl = 0
        onMassTtl = 0
        for d in docIDs:
            start = doc_range[d]
            stop = doc_range[d+1]
            words_d = word_id[start:stop]
            rstart = d * vocab_size
            rstop = (d+1) * vocab_size
            totalMassTtl += np.sum(respVec[rstart:rstop], axis=0)
            onMassTtl += np.sum(respVec[rstart + words_d], axis=0)
            aboveMask = respVec[rstart + words_d] >= minRespForEachTargetAtom
            massAbove = np.sum(
                respVec[rstart:rstop][aboveMask], axis=0)
            nAboveList.append(np.sum(aboveMask))
            massAboveList.append(massAbove)
        msg += "\n Total target respmass: %.2f, of which %.2f is ON (X=1)." % (
            totalMassTtl, onMassTtl)
        msg += "\n"
        msg += " Across target docs, distrib. of nTypes with X=1:"
        for p in [0, 10, 50, 90, 100]:
            md = np.percentile(nAboveList, p)
            msg += " %3d%% %5d  " % (p, md)
        msg += "\n"
        msg += "                              of respmass at X=1:"
        for p in [0, 10, 50, 90, 100]:
            md = np.percentile(massAboveList, p)
            msg += " %3d%% %7.1f" % (p, md)

    return msg

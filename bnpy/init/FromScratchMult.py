'''
FromScratchMult.py

Initialize params of an HModel with multinomial observations from scratch.
'''
from builtins import *
import numpy as np
import time
import os
import re
import warnings

from scipy.special import digamma
from .FromTruth import convertLPFromHardToSoft, convertLPFromDocsToTokens

# Import Kmeans routine
hasRexAvailable = True
try:
    import KMeansRex
except ImportError:
    hasRexAvailable = False

# Import spectral anchor-words routine
hasAnchorTopicEstimator = True
try:
    import AnchorTopicEstimator
except ImportError:
    hasAnchorTopicEstimator = False


def init_global_params(obsModel, Data, K=0, seed=0,
                       initname='randexamples',
                       initarg=None,
                       initMinWordsPerDoc=0,
                       **kwargs):
    ''' Initialize parameters for Mult obsModel, in place.

        Returns
        -------
        Nothing. obsModel is updated in place.
    '''
    PRNG = np.random.RandomState(seed)
    K = np.minimum(Data.nDoc, K)

    # Apply pre-processing to initialization Dataset
    # this removes documents with too few tokens, etc.
    if initMinWordsPerDoc > 0:
        targetDataArgs = dict(targetMinWordsPerDoc=initMinWordsPerDoc,
                              targetMaxSize=Data.nDoc,
                              targetMinSize=0,
                              randstate=PRNG)
        tmpData, tmpInfo = _sample_target_BagOfWordsData(Data, None, None,
                                                    **targetDataArgs)
        if tmpData is None:
            raise ValueError(
                'InitData preprocessing left no viable docs left.')
        Data = tmpData

    lam = None
    topics = None
    if initname == 'randomlikewang':
        # Sample K topics i.i.d. from Dirichlet with specified parameter
        # this method is exactly done in Chong Wang's onlinehdp code
        lam = PRNG.gamma(1.0, 1.0, (K, Data.vocab_size))
        lam *= Data.nDocTotal * 100.0 / (K * Data.vocab_size)
    else:
        topics, lam = _initTopicWordEstParams(obsModel, Data, PRNG,
                                         K=K,
                                         initname=initname,
                                         initarg=initarg,
                                         seed=seed,
                                         **kwargs)
    if hasattr(Data, 'clearCache'):
        Data.clearCache()
    InitArgs = dict(lam=lam, topics=topics, Data=Data)
    obsModel.set_global_params(**InitArgs)
    if 'savepath' in kwargs:
        import scipy.io
        topics = obsModel.getTopics()
        scipy.io.savemat(os.path.join(kwargs['savepath'], 'InitTopics.mat'),
                         dict(topics=topics), oned_as='row')

def _initTopicWordEstParams(obsModel, Data, PRNG, K=0,
                            initname='',
                            initarg='',
                            initObsModelAddRandomNoise=0,
                            initObsModelScale=0.0,
                            seed=0,
                            **kwargs):
    ''' Create initial guess for the topic-word parameter matrix

        Returns
        --------
        topics : 2D array, size K x Data.vocab_size
                 non-negative entries, rows sum to one
    '''
    topics = None
    lam = None

    extrafields = initname.split("+")
    initname = extrafields[0]
    for key in extrafields[1:]:
        m = re.match(
            r"(?P<name>[a-zA-Z]+)(?P<value>.+)$", key)
        name = m.group('name')
        value = m.group('value')
        if name.count("lam"):
            initObsModelScale = float(value)

    smoothParam = obsModel.Prior.lam[np.newaxis,:].copy()
    if initObsModelScale > 0.0:
        smoothParam += initObsModelScale

    if initname == 'randexamples':
        # Choose K documents at random, then
        # use each doc's empirical distribution (+random noise) to seed a topic
        K = np.minimum(K, Data.nDoc)
        chosenDocIDs = PRNG.choice(Data.nDoc, K, replace=False)
        DocWord = Data.getDocTypeCountMatrix()
        lam = DocWord[chosenDocIDs] + smoothParam

    elif initname == 'plusplus':
        # Sample K documents at random using 'plusplus' distance criteria
        # then set each of K topics to empirical distribution of chosen docs
        if not hasRexAvailable:
            raise NotImplementedError("KMeansRex must be on python path")
        K = np.minimum(K, Data.nDoc)
        X = Data.getDocTypeCountMatrix()
        lam = KMeansRex.SampleRowsPlusPlus(X, K, seed=seed)
        lam += smoothParam

    elif initname == 'kmeansplusplus':
        # Cluster all documents into K hard clusters via K-means
        # then set each of K topics to the means of the resulting clusters
        if not hasRexAvailable:
            raise NotImplementedError("KMeansRex must be on python path")
        K = np.minimum(K, Data.nDoc)
        X = Data.getDocTypeCountMatrix()
        lam, Z = KMeansRex.RunKMeans(X, K, seed=seed,
                                        Niter=25,
                                        initname='plusplus')
        for k in range(K):
            lam[k] = np.sum(X[Z == k], axis=0)
        lam += smoothParam

    elif initname == 'randomfromarg':
        # Draw K topic-word probability vectors i.i.d. from a Dirichlet
        # using user-provided symmetric parameter initarg
        topics = PRNG.gamma(initarg, 1., (K, Data.vocab_size))

    elif initname == 'randomfromprior':
        # Draw K topic-word probability vectors i.i.d. from their prior
        priorLam = obsModel.Prior.lam
        topics = PRNG.gamma(priorLam, 1., (K, Data.vocab_size))

    elif initname.count('anchor'):
        K = np.minimum(K, Data.vocab_size)

        # Set topic-word prob vectors to output of anchor-words spectral method
        if not hasAnchorTopicEstimator:
            raise NotImplementedError(
                "AnchorTopicEstimator must be on python path")

        stime = time.time()
        topics = AnchorTopicEstimator.findAnchorTopics(
            Data, K, seed=seed,
            minDocPerWord=kwargs['initMinDocPerWord'],
            lowerDim=kwargs['initDim'])
        elapsedtime = time.time() - stime
        assert np.allclose(topics.sum(axis=1), 1.0)
    else:
        raise NotImplementedError('Unrecognized initname ' + initname)
    # .... end initname switch

    if lam is not None:
        if np.any(np.isnan(lam.sum(axis=1))):
            raise ValueError("NaN")
        if initObsModelAddRandomNoise:
            lam += 0.1 * smoothParam * PRNG.rand(lam.shape[0], lam.shape[1])

    if topics is None and obsModel.inferType.count('EM'):
        topics = lam / lam.sum(axis=1)[:, np.newaxis]

    if topics is not None:
        # Double-check for suspicious NaN values
        # These can arise if kmeans delivers any empty clusters
        rowSum = topics.sum(axis=1)
        mask = np.isnan(rowSum)
        if np.any(mask):
            warnings.warn("%d topics had NaN values. Filled with random noize."
                % (np.sum(mask)))
            # Fill in any bad rows with uniform noise
            topics[mask] = PRNG.rand(np.sum(mask), Data.vocab_size)
        np.maximum(topics, 1e-100, out=topics)
        topics /= topics.sum(axis=1)[:, np.newaxis]

        # Raise error if any NaN detected
        if np.any(np.isnan(topics)):
            raise ValueError('topics should never be NaN')
        assert np.allclose(np.sum(topics, axis=1), 1.0)
    return topics, lam

def _sample_target_BagOfWordsData(Data, model, LP, return_Info=0, **kwargs):
    ''' Get subsample of set of documents satisfying provided criteria.

    minimum size of each document, relationship to targeted component, etc.

    Keyword Args
    --------
    targetCompID : int, range: [0, 1, ... K-1]. **optional**
                 if present, we target documents that use a specific topic

    targetMinWordsPerDoc : int,
                         each document in returned targetData
                         must have at least this many words
    Returns
    --------
    targetData : BagOfWordsData dataset,
                with at most targetMaxSize documents
    DebugInfo : (optional), dictionary with debugging info
    '''
    DocWordMat = Data.getSparseDocTypeCountMatrix()
    DebugInfo = dict()

    candidates = np.arange(Data.nDoc)
    if kwargs['targetMinWordsPerDoc'] > 0:
        nWordPerDoc = np.asarray(DocWordMat.sum(axis=1))
        candidates = nWordPerDoc >= kwargs['targetMinWordsPerDoc']
        candidates = np.flatnonzero(candidates)
    if len(candidates) < 1:
        return None, dict()

    # ............................................... target a specific Comp
    if hasValidKey('targetCompID', kwargs):
        if hasValidKey('DocTopicCount', LP):
            Ndk = LP['DocTopicCount'][candidates].copy()
            Ndk /= np.sum(Ndk, axis=1)[:, np.newaxis] + 1e-9
            mask = Ndk[:, kwargs['targetCompID']] > kwargs['targetCompFrac']
        elif hasValidKey('resp', LP):
            mask = LP['resp'][
                :,
                kwargs['targetCompID']] > kwargs['targetCompFrac']
            if candidates is not None:
                mask = mask[candidates]
        else:
            raise ValueError('LP must have either DocTopicCount or resp')
        candidates = candidates[mask]

    # ............................................... target a specific Word
    elif hasValidKey('targetWordIDs', kwargs):
        wordIDs = kwargs['targetWordIDs']
        TinyMatrix = DocWordMat[candidates, :].toarray()[:, wordIDs]
        targetCountPerDoc = np.sum(TinyMatrix > 0, axis=1)
        mask = targetCountPerDoc >= kwargs['targetWordMinCount']
        candidates = candidates[mask]

    # ............................................... target based on WordFreq
    elif hasValidKey('targetWordFreq', kwargs):
        wordFreq = kwargs['targetWordFreq']
        from TargetPlannerWordFreq import calcDocWordUnderpredictionScores

        ScoreMat = calcDocWordUnderpredictionScores(Data, model, LP)
        ScoreMat = ScoreMat[candidates]
        DebugInfo['ScoreMat'] = ScoreMat
        if kwargs['targetSelectName'].count('score'):
            ScoreMat = np.maximum(0, ScoreMat)
            ScoreMat /= ScoreMat.sum(axis=1)[:, np.newaxis]
            distPerDoc = calcDistBetweenHist(ScoreMat, wordFreq)

            DebugInfo['distPerDoc'] = distPerDoc
        else:
            EmpWordFreq = DocWordMat[candidates, :].toarray()
            EmpWordFreq /= EmpWordFreq.sum(axis=1)[:, np.newaxis]
            distPerDoc = calcDistBetweenHist(EmpWordFreq, wordFreq)
            DebugInfo['distPerDoc'] = distPerDoc

        keepIDs = distPerDoc.argsort()[:kwargs['targetMaxSize']]
        candidates = candidates[keepIDs]
        DebugInfo['candidates'] = candidates
        DebugInfo['dist'] = distPerDoc[keepIDs]

    if len(candidates) < 1:
        return None, dict()
    elif len(candidates) <= kwargs['targetMaxSize']:
        targetData = Data.make_subset(candidates)
    else:
        targetData = Data.get_random_sample(kwargs['targetMaxSize'],
                                            randstate=kwargs['randstate'],
                                            candidates=candidates)

    return targetData, DebugInfo

def hasValidKey(key, kwargs):
    return key in kwargs and kwargs[key] is not None


"""

def initSSByBregDiv_Mult(
        Dslice=None,
        curModel=None,
        curLPslice=None,
        K=5, ktarget=None,
        b_minNumAtomsInDoc=None,
        b_includeRemainderTopic=0,
        b_initHardCluster=0,
        seed=0, doSample=True,
        **kwargs):
    ''' Create observation model statistics via Breg. distance sampling.

    Returns
    -------
    xSS : SuffStatBag
    Info : dict
        contains info about which docs were used to inspire this init.
    '''
    PRNG = np.random.RandomState(seed)
    if curLPslice is None:
        weights = None
    else:
        weights = curLPslice['resp'][:,ktarget]
    # Make nDoc x vocab_size array
    DocWordMat = Dslice.getDocTypeCountMatrix(weights=weights)
    # Keep only rows with minimum count
    if b_minNumAtomsInDoc is None:
        rowsWithEnoughData = np.arange(DocWordMat.shape[0])
    else:
        rowsWithEnoughData = np.flatnonzero(
            DocWordMat.sum(axis=1) > b_minNumAtomsInDoc)
    enoughDocWordMat = DocWordMat[rowsWithEnoughData]

    Keff = np.minimum(K, enoughDocWordMat.shape[0])
    if Keff < 1:
        DebugInfo = dict(
            msg="Not enough data. Looked for %d documents, found only %d." % (
                K, Keff))
        return None, DebugInfo

    K = Keff
    np.maximum(enoughDocWordMat, 1e-100, out=enoughDocWordMat)

    WholeDataMean = calcClusterMean_Mult(
        enoughDocWordMat, hmodel=curModel)[np.newaxis, :]
    minDiv = calcBregDiv_Mult(enoughDocWordMat, WholeDataMean)
    assert minDiv.min() > -1e-10
    WCMeans = np.zeros((K, WholeDataMean.shape[1]))
    lamVals = np.zeros(K+1)
    chosenDocIDs = np.zeros(K, dtype=np.int32)
    for k in range(K):
        # Find data point with largest minDiv value
        if doSample:
            pvec = minDiv[:,0] / np.sum(minDiv)
            n = PRNG.choice(minDiv.size, p=pvec)
        else:
            n = minDiv.argmax()
        chosenDocIDs[k] = rowsWithEnoughData[n]
        lamVals[k] = minDiv[n]
        # Add this point to the clusters
        WCMeans[k,:] = calcClusterMean_Mult(
            enoughDocWordMat[n], hmodel=curModel)
        # Recalculate minimum distance to existing means
        curDiv = calcBregDiv_Mult(enoughDocWordMat, WCMeans[k])
        np.minimum(curDiv, minDiv, out=minDiv)
        minDiv[n] = 0
        assert minDiv.min() > -1e-10
    lamVals[-1] = minDiv.max()

    if b_includeRemainderTopic == 1:
        chosenDocIDs = chosenDocIDs[:-1]
        WCMeans = np.vstack([WholeDataMean, WCMeans[:-1]])
        WCMeans[0] -= DocWordMat[chosenDocIDs].sum(axis=0)

    Z = -1 * np.ones(Dslice.nDoc, dtype=np.int32)
    if b_initHardCluster:
        DivMat = calcBregDiv_Mult(enoughDocWordMat, WCMeans)
        Z[rowsWithEnoughData] = DivMat.argmin(axis=1)
    else:
        if b_includeRemainderTopic:
            Z[chosenDocIDs] = 1 + np.arange(len(chosenDocIDs))
            Z[Z<1] = 0 # all other docs to rem cluster
        else:
            Z[chosenDocIDs] = np.arange(len(chosenDocIDs))

    xdocLP = convertLPFromHardToSoft(
        dict(Z=Z), Dslice, initGarbageState=0)
    if curModel.obsModel.DataAtomType.count('word'):
        xLP = convertLPFromDocsToTokens(xdocLP, Dslice)
    else:
        xLP = xdocLP
    if curLPslice is not None:
        xLP['resp'] *= curLPslice['resp'][:, ktarget][:,np.newaxis]

        # Verify that initial xLP resp is a subset of curLP's resp,
        # leaving out only the docs that didnt have enough tokens.
        assert np.all(xLP['resp'].sum(axis=1) <= \
                      curLPslice['resp'][:, ktarget] + 1e-5)
    xSS = curModel.obsModel.get_global_suff_stats(Dslice, None, xLP)

    # Reorder the components from big to small
    bigtosmall = np.argsort(-1 * xSS.getCountVec())
    xSS.reorderComps(bigtosmall)
    Info = dict(
        Z=Z,
        Means=WCMeans,
        lamVals=lamVals,
        chosenDocIDs=chosenDocIDs)
    return xSS, Info

def calcClusterMean_Mult(WordCountData, lam=0.05, hmodel=None):
    if hmodel is not None:
        lam = hmodel.obsModel.Prior.lam
    if WordCountData.ndim == 1:
        WordCountData = WordCountData[np.newaxis,:]
    WordCountSumVec = np.sum(WordCountData, axis=0)
    ClusterMean = WordCountSumVec + lam
    return ClusterMean

def calcBregDiv_Mult(WordCountData, WordCountMeans):
    ''' Calculate Bregman divergence between rows of two matrices.

    Args
    ----
    WordCountData : 2D array, N x vocab_size
    WordCountMeans : 2D array, K x vocab_size

    Returns
    -------
    Div : 2D array, N x K
    '''
    if WordCountData.ndim == 1:
        WordCountData = WordCountData[np.newaxis,:]
    if WordCountMeans.ndim == 1:
        WordCountMeans = WordCountMeans[np.newaxis,:]
    assert WordCountData.shape[1] == WordCountMeans.shape[1]
    N = WordCountData.shape[0]
    K = WordCountMeans.shape[0]
    Nx = WordCountData.sum(axis=1)
    assert np.all(Nx >= 1.0 - 1e-10)
    Nmean = WordCountMeans.sum(axis=1)
    assert np.all(Nmean >= 1.0 - 1e-10)
    Div = np.zeros((N, K))
    for k in xrange(K):
        Div[:, k] = np.sum(WordCountData * np.log(
            WordCountData / WordCountMeans[k,:][np.newaxis,:]), axis=1)
        Div[:, k] += Nx * np.log(Nmean[k]/Nx)
    return Div
"""

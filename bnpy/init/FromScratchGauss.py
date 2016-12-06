'''
FromScratchGauss.py

Initialize global params of a Gaussian-family data-generation model,
from scratch.
'''

import numpy as np
from bnpy.data import XData
from bnpy.suffstats import SuffStatBag
from scipy.cluster.vq import kmeans2
from FromTruth import convertLPFromHardToSoft

def init_global_params(obsModel, Data, K=0, seed=0,
                       initname='randexamples',
                       initBlockLen=20,
                       **kwargs):
    ''' Initialize parameters for Gaussian obsModel, in place.

    Parameters
    -------
    obsModel : bnpy.obsModel subclass
        Observation model object to initialize.
    Data : bnpy.data.DataObj
        Dataset to use to drive initialization.
        obsModel dimensions must match this dataset.
    initname : str
        name of routine used to do initialization
        Options: ['randexamples', 'randexamplesbydist', 'kmeans',
                  'randcontigblocks', 'randsoftpartition',
                 ]

    Post Condition
    -------
    obsModel has valid global parameters.
    Either its EstParams or Post attribute will be contain K components.
    '''
    K = int(K)
    PRNG = np.random.RandomState(seed)
    X = Data.X
    if initname == 'randexamples':
        # Choose K items uniformly at random from the Data
        #    then component params by M-step given those single items
        resp = np.zeros((Data.nObs, K))
        permIDs = PRNG.permutation(Data.nObs).tolist()
        for k in xrange(K):
            resp[permIDs[k], k] = 1.0

    elif initname == 'randexamplesbydist':
        # Choose K items from the Data,
        #  selecting the first at random,
        # then subsequently proportional to euclidean distance to the closest
        # item
        objID = PRNG.choice(Data.nObs)
        chosenObjIDs = list([objID])
        minDistVec = np.inf * np.ones(Data.nObs)
        for k in range(1, K):
            curDistVec = np.sum((Data.X - Data.X[objID])**2, axis=1)
            minDistVec = np.minimum(minDistVec, curDistVec)
            objID = PRNG.choice(Data.nObs, p=minDistVec / minDistVec.sum())
            chosenObjIDs.append(objID)
        resp = np.zeros((Data.nObs, K))
        for k in xrange(K):
            resp[chosenObjIDs[k], k] = 1.0

    elif initname == 'randcontigblocks':
        # Choose K contig blocks of provided size from the Data,
        #  selecting each block at random from a particular sequence
        if hasattr(Data, 'doc_range'):
            doc_range = Data.doc_range.copy()
        else:
            doc_range = np.asarray([0, Data.X.shape[0]])
        nDoc = doc_range.size - 1
        docIDs = np.arange(nDoc)
        PRNG.shuffle(docIDs)
        resp = np.zeros((Data.nObs, K))
        for k in xrange(K):
            n = docIDs[k % nDoc]
            start = doc_range[n]
            stop = doc_range[n + 1]
            T = stop - start
            if initBlockLen >= T:
                a = start
                b = stop
            else:
                a = start + PRNG.choice(T - initBlockLen)
                b = a + initBlockLen
            resp[a:b, k] = 1.0

    elif initname == 'randsoftpartition':
        # Randomly assign all data items some mass in each of K components
        #  then create component params by M-step given that soft partition
        resp = PRNG.gamma(1.0 / (K * K), 1, size=(Data.nObs, K))
        resp[resp < 1e-3] = 0
        rsum = np.sum(resp, axis=1)
        badIDs = rsum < 1e-8
        # if any rows have no content, just set them to unif resp.
        if np.any(badIDs):
            resp[badIDs] = 1.0 / K
            rsum[badIDs] = 1
        resp = resp / rsum[:, np.newaxis]
        assert np.allclose(np.sum(resp, axis=1), 1.0)

    elif initname == 'kmeans':
        # Fill in resp matrix with hard-clustering from K-means
        # using an initialization with K randomly selected points from X
        np.random.seed(seed)
        centroids, labels = kmeans2(data=Data.X, k=K, minit='points')
        resp = np.zeros((Data.nObs, K))
        for t in xrange(Data.nObs):
            resp[t, labels[t]] = 1

    else:
        raise NotImplementedError('Unrecognized initname ' + initname)

    tempLP = dict(resp=resp)
    SS = SuffStatBag(K=K, D=Data.dim)
    SS = obsModel.get_global_suff_stats(Data, SS, tempLP)
    obsModel.update_global_params(SS)


def initSSByBregDiv_ZeroMeanGauss(
        Dslice=None, 
        curModel=None, 
        curLPslice=None,
        K=5, ktarget=None, 
        b_minRespToIncludeInInit=None, 
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
        targetAtoms = np.arange(Dslice.nObs)
        targetX = Dslice.X
    else:
        targetAtoms = np.flatnonzero(
            curLPslice['resp'][:,ktarget] > b_minRespToIncludeInInit)
        targetX = Dslice.X[targetAtoms]

    Keff = np.minimum(K, targetX.shape[0])
    if Keff < 1:
        DebugInfo = dict(
            msg="Not enough data. Looked for %d atoms, found only %d." % (
                K, Keff))
        return None, DebugInfo

    K = Keff
    WholeDataMean = calcClusterMean_ZeroMeanGauss(
        targetX, hmodel=curModel)
    Mu = np.zeros((K, Dslice.dim, Dslice.dim))    
    minDiv = np.inf * np.ones((targetX.shape[0],1))
    lamVals = np.zeros(K)
    chosenAtomIDs = np.zeros(K, dtype=np.int32)
    for k in range(K):
        if k == 0:
            # Choose first point uniformly at randomly
            n = PRNG.choice(minDiv.size)
        else:
            if doSample:
                pvec = minDiv[:,0] / np.sum(minDiv)
                n = PRNG.choice(minDiv.size, p=pvec)
            else:
                n = minDiv.argmax()
        chosenAtomIDs[k] = targetAtoms[n]
        # Add this point to the clusters
        Mu[k] = calcClusterMean_ZeroMeanGauss(
            targetX[n], hmodel=curModel)
        # Recalculate minimum distance to existing means
        curDiv = calcBregDiv_ZeroMeanGauss(targetX, Mu[k])
        np.minimum(curDiv, minDiv, out=minDiv)
        lamVals[k] = minDiv[n]
        minDiv[n] = 1e-10
        minDiv = np.maximum(minDiv, 1e-10)
        assert minDiv.min() > -1e-10
    
    Z = -1 * np.ones(Dslice.nObs, dtype=np.int32)
    if b_initHardCluster:
        DivMat = calcBregDiv_ZeroMeanGauss(targetX, Mu)
        Z[targetAtoms] = DivMat.argmin(axis=1)
    else:
        Z[chosenAtomIDs] = np.arange(len(chosenAtomIDs))

    xLP = convertLPFromHardToSoft(
        dict(Z=Z), Dslice, initGarbageState=0)
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
        Mu=Mu, 
        lamVals=lamVals, 
        chosenAtomIDs=chosenAtomIDs)
    return xSS, Info

def calcClusterMean_ZeroMeanGauss(X, hmodel=None):
    ''' Compute MAP value of Mu given provided data X

    Returns
    -------
    Mu : 2D array, D x D
    '''
    if hmodel is not None:
        B = hmodel.obsModel.Prior.B
        nu = hmodel.obsModel.Prior.nu
    if X.ndim == 1:
        X = X[np.newaxis,:]
    Mu = (np.dot(X.T, X) + B) / (nu + X.shape[0])
    return Mu

def calcBregDiv_ZeroMeanGauss(X, Mu):
    ''' Calculate Bregman divergence between rows of two matrices.

    Args
    ----
    X : 2D array, N x D
    Mu : 2D array, K x D x D

    Returns
    -------
    Div : 2D array, N x K
    '''
    if X.ndim == 1:
        X = X[np.newaxis,:]
    if Mu.ndim == 2:
        Mu = Mu[np.newaxis,:]
    assert Mu.ndim == 3
    assert X.shape[1] == Mu.shape[2]
    N, D = X.shape
    K = Mu.shape[0]
    Div = np.zeros((N, K))
    logdetX = np.log(np.square(X) + 1e-100).sum(axis=1)
    for k in xrange(K):
        # cholMu_k is a lower-triangular matrix
        cholMu_k = np.linalg.cholesky(Mu[k])
        logdetMu_k = 2 * np.sum(np.log(np.diag(cholMu_k)))
        xxTinvMu = np.linalg.solve(cholMu_k, X.T)
        xxTinvMu *= xxTinvMu
        tr_xxTinvMu = np.sum(xxTinvMu, axis=0)
        Div[:,k] = - 0.5 * D + 0.5 * tr_xxTinvMu
        # Div[:, k] += 0.5 * logdetMu_k - 0.5 * logdetX
        assert Div.min() > -1e-10
    return Div

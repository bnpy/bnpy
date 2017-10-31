'''
FromScratchBern.py

Initialize global params of Bernoulli data-generation model,
from scratch.
'''
from builtins import *
import numpy as np
from bnpy.data import XData, BagOfWordsData
from bnpy.suffstats import SuffStatBag
from scipy.cluster.vq import kmeans2


def init_global_params(obsModel, Data, K=0, seed=0,
                       initname='randexamples',
                       initBlockLen=20,
                       **kwargs):
    ''' Initialize parameters for Bernoulli obsModel, in place.

    Parameters
    -------
    obsModel : bnpy.obsModel subclass
        Observation model object to initialize.
    Data   : bnpy.data.DataObj
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
    PRNG = np.random.RandomState(seed)
    if hasattr(Data, 'X'):
        X = Data.X
        N = X.shape[0]
        D = X.shape[1]
    elif isinstance(Data, BagOfWordsData):
        X = Data.getDocTypeBinaryMatrix()
        N = X.shape[0]
        D = X.shape[1]
    if initname == 'randexamples':
        # Choose K items uniformly at random from the Data
        #    then component params by M-step given those single items
        resp = np.zeros((N, K))
        permIDs = PRNG.permutation(N).tolist()
        for k in range(K):
            resp[permIDs[k], k] = 1.0

    elif initname == 'randexamplesbydist':
        # Choose K items from the Data,
        #  selecting the first at random,
        # then subsequently proportional to euclidean distance to the closest
        # item
        K = np.minimum(K, N)
        objID = PRNG.choice(N)
        chosenObjIDs = list([objID])
        minDistVec = np.inf * np.ones(N)
        for k in range(1, K):
            curDistVec = np.sum((X - X[objID])**2, axis=1)
            minDistVec = np.minimum(minDistVec, curDistVec)
            sum_minDistVec = np.sum(minDistVec)
            if sum_minDistVec > 0:
                p = minDistVec / sum_minDistVec
            else:
                DD = minDistVec.size
                p = 1.0 / DD * np.ones(DD)
            objID = PRNG.choice(N, p=p)
            chosenObjIDs.append(objID)
        resp = np.zeros((N, K))
        for k in range(K):
            resp[chosenObjIDs[k], k] = 1.0

    elif initname == 'randcontigblocks':
        # Choose K contig blocks of provided size from the Data,
        #  selecting each block at random from a particular sequence
        if hasattr(Data, 'doc_range'):
            doc_range = Data.doc_range.copy()
        else:
            doc_range = [0, N]
        nDoc = doc_range.size - 1
        docIDs = np.arange(nDoc)
        PRNG.shuffle(docIDs)
        resp = np.zeros((N, K))
        for k in range(K):
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

    elif initname == 'kmeans':
        # Fill in resp matrix with hard-clustering from K-means
        # using an initialization with K randomly selected points from X
        np.random.seed(seed)
        centroids, labels = kmeans2(data=X, k=K, minit='points')
        resp = np.zeros((N, K))
        for n in range(N):
            resp[n, labels[n]] = 1

    elif initname == 'randsoftpartition':
        # Randomly assign all data items some mass in each of K components
        #  then create component params by M-step given that soft partition
        resp = PRNG.gamma(1.0 / (K * K), 1, size=(N, K))
        resp[resp < 1e-3] = 0
        rsum = np.sum(resp, axis=1)
        badIDs = rsum < 1e-8
        # if any rows have no content, just set them to unif resp.
        if np.any(badIDs):
            resp[badIDs] = 1.0 / K
            rsum[badIDs] = 1
        resp = resp / rsum[:, np.newaxis]
        assert np.allclose(np.sum(resp, axis=1), 1.0)

    else:
        raise NotImplementedError('Unrecognized initname ' + initname)

    # Using the provided resp for each token,
    # we summarize into sufficient statistics
    # then perform one global step (M step) to get initial global params
    tempLP = dict(resp=resp)
    SS = SuffStatBag(K=K, D=Data.dim)
    SS = obsModel.get_global_suff_stats(Data, SS, tempLP)
    obsModel.update_global_params(SS)

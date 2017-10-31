from builtins import *
import sys
import os
import scipy.sparse
import numpy as np
from bnpy.util import argsort_bigtosmall_stable


def loadKeffForTask(
        taskpath,
        effCountThr=0.01,
        MIN_PRESENT_COUNT=1e-10,
        **kwargs):
    ''' Load effective number of clusters used at each checkpoint.

    Returns
    -------
    Keff : 1D array, size nCheckpoint
    '''
    effCountThr = np.maximum(effCountThr, MIN_PRESENT_COUNT)
    CountMat, Info = loadCountHistoriesForTask(taskpath,
        MIN_PRESENT_COUNT=MIN_PRESENT_COUNT)
    return np.sum(CountMat >= effCountThr, axis=1)

def loadCountHistoriesForTask(
        taskpath,
        sortBy=None,
        MIN_PRESENT_COUNT=1e-10):
    ''' Load sparse matrix of counts for all clusters used throughout task.

    Returns
    -------
    AllCountMat : 2D array, nCheckpoint x nTotal
    Info : dict
    '''
    idpath = os.path.join(taskpath, 'ActiveIDs.txt')
    ctpath = os.path.join(taskpath, 'ActiveCounts.txt')
    fid = open(idpath, 'r')
    fct = open(ctpath, 'r')
    data = list()
    colids = list()
    rowids = list()
    for ii, idline in enumerate(fid.readlines()):
        idstr = str(idline.strip())
        ctstr = str(fct.readline().strip())
        idvec = np.asarray(idstr.split(' '), dtype=np.int32)
        ctvec = np.asarray(ctstr.split(' '), dtype=np.float)
        data.extend(ctvec)
        colids.extend(idvec)
        rowids.extend( ii * np.ones(idvec.size))

    # Identify columns by unique ids
    allUIDs = np.unique(colids)
    compactColIDs = -1 * np.ones_like(colids)
    for pos, u in enumerate(allUIDs):
        mask = colids == u
        compactColIDs[mask] = pos
    assert compactColIDs.min() >= 0

    # CountMat : sparse matrix of active counts at each checkpoint
    # Each row gives count (or zero if eliminated) at single lap
    data = np.asarray(data)
    np.maximum(data, MIN_PRESENT_COUNT, out=data)
    ij = np.vstack([rowids, compactColIDs])
    CountMat = scipy.sparse.csr_matrix((data, ij))
    CountMat = CountMat.toarray()
    assert allUIDs.size == CountMat.shape[1]

    # Split all columns into two sets: active and eliminated
    nCol = CountMat.shape[1]
    elimCols = np.flatnonzero(CountMat[-1, :] < MIN_PRESENT_COUNT)
    activeCols = np.setdiff1d(np.arange(nCol), elimCols)
    nElimCol = len(elimCols)
    nActiveCol = len(activeCols)
    ElimCountMat = CountMat[:, elimCols]
    ActiveCountMat = CountMat[:, activeCols]
    elimUIDs = allUIDs[elimCols]
    activeUIDs = allUIDs[activeCols]

    # Fill out info dict
    Info = dict(
        CountMat=CountMat,
        allUIDs=allUIDs,
        ActiveCountMat=ActiveCountMat,
        ElimCountMat=ElimCountMat,
        activeCols=activeCols,
        elimCols=elimCols,
        activeUIDs=activeUIDs,
        elimUIDs=elimUIDs)

    if not isinstance(sortBy, str) or sortBy.lower().count('none'):
        return CountMat, Info

    if sortBy.lower().count('finalorder'):
        rankedActiveUIDs = idvec
        raise ValueError("TODO")
    elif sortBy.lower().count('countvalues'):
        ## Sort columns from biggest to smallest (at last chkpt)
        rankedActiveIDs = argsort_bigtosmall_stable(ActiveCountMat[-1,:])
    else:
        raise ValueError("TODO")

    # Sort active set by size at last snapshot
    ActiveCountMat = ActiveCountMat[:, rankedActiveIDs]
    activeUIDs = activeUIDs[rankedActiveIDs]
    activeCols = activeCols[rankedActiveIDs]

    # Sort eliminated set by historical size
    rankedElimIDs = argsort_bigtosmall_stable(ElimCountMat.sum(axis=0))
    ElimCountMat = ElimCountMat[:, rankedElimIDs]
    elimUIDs = elimUIDs[rankedElimIDs]
    elimCols = elimCols[rankedElimIDs]

    Info['activeUIDs'] = activeUIDs
    Info['activeCols'] = activeCols
    Info['elimUIDs'] = elimUIDs
    Info['elimCols'] = elimCols
    return ActiveCountMat, ElimCountMat, Info

def LoadActiveIDsForTaskFromLap(taskpath, queryLap='final'):
    ''' Load vector of active cluster UIDs for specific lap

    Essentially reads a single line of the ActiveIDs.txt file from taskpath

    Returns
    -------
    idvec : 1D array, size K
        where K is number of clusters active at chosen lap
    '''
    lappath = os.path.join(taskpath, 'laps.txt')
    laps = np.loadtxt(lappath)
    if queryLap is not None and queryLap != 'final':
        if queryLap not in laps:
            raise ValueError('Target lap not found.')
    idpath = os.path.join(taskpath, 'ActiveIDs.txt')
    with open(idpath, 'r') as f:
        for ii, curLap in enumerate(laps):
            idstr = f.readline().strip()
            if curLap == queryLap or (curLap == laps[-1] and queryLap == 'final'):
                idvec = np.asarray(idstr.split(' '), dtype=np.int32)
                return idvec



if __name__ == '__main__':
    tpath = "/data/liv/xdump/BerkPatchB1/billings-alg=bnpyHDPbirthmerge-lik=ZeroMeanGauss-ECovMat=diagcovdata-sF=0.1-K=1-initname=bregmankmeans-nBatch=1/1/"
    loadCountHistoriesForTask(tpath)

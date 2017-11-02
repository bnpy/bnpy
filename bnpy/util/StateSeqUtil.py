from __future__ import print_function
from builtins import *
import numpy as np
try:
    import munkres
    hasMunkres = True
except ImportError as e:
    hasMunkres = False

from bnpy.util import as1D


def calcHammingDistance(zTrue, zHat, excludeNegLabels=1, verbose=0,
                        **kwargs):
    ''' Compute Hamming distance: sum of all timesteps with different labels.

    Normalizes result to be within [0, 1].

    Args
    --------
    zHat : 1D array
        each entry is an integer label in {0, 1, ... Kest-1}
    zTrue : 1D array
        each entry is an integer label in {0, 1, ... Ktrue-1}

    Returns
    ------
    d : int
        Hamming distance from zTrue to zHat.

    Examples
    ------
    >>> calcHammingDistance([0, 0, 1, 1], [0, 0, 1, 1])
    0.0
    >>> calcHammingDistance([0, 0, 1, 1], [0, 0, 1, 2])
    0.25
    >>> calcHammingDistance([0, 0, 1, 1], [1, 1, 0, 0])
    1.0
    >>> calcHammingDistance([1, 1, 0, -1], [1, 1, 0, 0])
    0.0
    >>> calcHammingDistance([-1, -1, -2, 3], [1, 2, 3, 3])
    0.0
    >>> calcHammingDistance([-1, -1, 0, 1], [1, 2, 0, 1], excludeNegLabels=1)
    0.0
    >>> calcHammingDistance([-1, -1, 0, 1], [1, 2, 0, 1], excludeNegLabels=0)
    0.5
    '''
    zHat = as1D(zHat)
    zTrue = as1D(zTrue)
    if excludeNegLabels:
        assert np.sum(zHat < 0) == 0
        good_tstep_mask = zTrue >= 0
        nGood = np.sum(good_tstep_mask)
        if verbose and np.sum(good_tstep_mask) < zTrue.size:
            print('EXCLUDED %d/%d timesteps' % (np.sum(zTrue < 0), zTrue.size))
        dist = np.sum(zTrue[good_tstep_mask] != zHat[good_tstep_mask])
        dist = dist/float(nGood)
    else:
        dist = np.sum(zTrue != zHat) / float(zHat.size)
    return dist


def buildCostMatrix(zHat, zTrue):
    ''' Construct cost matrix for alignment of estimated and true sequences

    Args
    --------
    zHat : 1D array
        each entry is an integer label in {0, 1, ... Kest-1}
    zTrue : 1D array
        each entry is an integer label in {0, 1, ... Ktrue-1}
        with optional negative state labels

    Returns
    --------
    CostMatrix : 2D array, size Ktrue x Kest
        CostMatrix[j,k] = count of events across all timesteps,
        where j is assigned, but k is not.
    '''
    zHat = as1D(zHat)
    zTrue = as1D(zTrue)
    Ktrue = int(np.max(zTrue)) + 1
    Kest = int(np.max(zHat)) + 1
    K = np.maximum(Ktrue, Kest)
    CostMatrix = np.zeros((K, K))
    for ktrue in range(K):
        for kest in range(K):
            CostMatrix[ktrue, kest] = np.sum(np.logical_and(zTrue == ktrue,
                                                            zHat != kest))
    return CostMatrix


def alignEstimatedStateSeqToTruth(zHat, zTrue, useInfo=None, returnInfo=False):
    ''' Relabel the states in zHat to minimize the hamming-distance to zTrue

    Args
    --------
    zHat : 1D array
        each entry is an integer label in {0, 1, ... Kest-1}
    zTrue : 1D array
        each entry is an integer label in {0, 1, ... Ktrue-1}

    Returns
    --------
    zHatAligned : 1D array
        relabeled version of zHat that aligns to zTrue
    AInfo : dict
        information about the alignment
    '''
    zHat = as1D(zHat)
    zTrue = as1D(zTrue)
    Kest = zHat.max() + 1
    Ktrue = zTrue.max() + 1

    if useInfo is None:
        if not hasMunkres:
            raise ImportError(
                "alignEstimatedStateSeqToTruth requires the munkres package."
                + " Please install via 'pip install munkres'")

        CostMatrix = buildCostMatrix(zHat, zTrue)
        MunkresAlg = munkres.Munkres()
        tmpAlignedRowColPairs = MunkresAlg.compute(CostMatrix)
        AlignedRowColPairs = list()
        OrigToAlignedMap = dict()
        AlignedToOrigMap = dict()
        for (ktrue, kest) in tmpAlignedRowColPairs:
            if kest < Kest:
                AlignedRowColPairs.append((ktrue, kest))
                OrigToAlignedMap[kest] = ktrue
                AlignedToOrigMap[ktrue] = kest
    else:
        # Unpack existing alignment info
        AlignedRowColPairs = useInfo['AlignedRowColPairs']
        CostMatrix = useInfo['CostMatrix']
        AlignedToOrigMap = useInfo['AlignedToOrigMap']
        OrigToAlignedMap = useInfo['OrigToAlignedMap']
        Ktrue = useInfo['Ktrue']
        Kest = useInfo['Kest']

        assert np.allclose(Ktrue, zTrue.max() + 1)
        Khat = zHat.max() + 1

        # Account for extra states present in zHat
        # that have never been aligned before.
        # They should align to the next available UID in set
        # [Ktrue, Ktrue+1, Ktrue+2, ...]
        # so they don't get confused for a true label
        ktrueextra = np.max([r for r, c in AlignedRowColPairs])
        ktrueextra = int(np.maximum(ktrueextra + 1, Ktrue))
        for khat in np.arange(Kest, Khat + 1):
            if khat in OrigToAlignedMap:
                continue
            OrigToAlignedMap[khat] = ktrueextra
            AlignedToOrigMap[ktrueextra] = khat
            AlignedRowColPairs.append((ktrueextra, khat))
            ktrueextra += 1

    zHatA = -1 * np.ones_like(zHat)
    for kest in np.unique(zHat):
        mask = zHat == kest
        zHatA[mask] = OrigToAlignedMap[kest]
    assert np.all(zHatA >= 0)

    if returnInfo:
        return zHatA, dict(CostMatrix=CostMatrix,
                           AlignedRowColPairs=AlignedRowColPairs,
                           OrigToAlignedMap=OrigToAlignedMap,
                           AlignedToOrigMap=AlignedToOrigMap,
                           Ktrue=Ktrue,
                           Kest=Kest)
    else:
        return zHatA


def convertStateSeq_flat2list(zFlat, Data):
    ''' Convert flat, 1D array representation of multiple sequences to list
    '''
    zListBySeq = list()
    for n in range(Data.nDoc):
        start = Data.doc_range[n]
        stop = Data.doc_range[n + 1]
        zListBySeq.append(zFlat[start:stop])
    return zListBySeq


def convertStateSeq_list2flat(zListBySeq, Data):
    ''' Convert nested list representation of multiple sequences to 1D array
    '''
    zFlat = np.zeros(Data.doc_range[-1])
    for n in range(Data.nDoc):
        start = Data.doc_range[n]
        stop = Data.doc_range[n + 1]
        zFlat[start:stop] = zListBySeq[n]
    return zFlat


def convertStateSeq_list2MAT(zListBySeq):
    ''' Convert nested list representation to MAT friendly format
    '''
    N = len(zListBySeq)
    zObjArr = np.zeros((N, 1), dtype=object)
    for n in range(N):
        zObjArr[n, 0] = np.asarray(
            zListBySeq[n][:, np.newaxis], dtype=np.int32)
    return zObjArr


def convertStateSeq_MAT2list(zObjArr):
    N = zObjArr.shape[0]
    zListBySeq = list()
    for n in range(N):
        zListBySeq.append(np.squeeze(zObjArr[n, 0]))
    return zListBySeq


def calcContigBlocksFromZ(Zvec, returnStates=False):
    ''' Identify contig blocks assigned to one state in Zvec

    Examples
    --------
    >>> calcContigBlocksFromZ([0,0,0,0])
    (array([ 4.]), array([ 0.]))
    >>> calcContigBlocksFromZ([0,0,0,1,1])
    (array([ 3.,  2.]), array([ 0.,  3.]))
    >>> calcContigBlocksFromZ([0,1,0])
    (array([ 1.,  1.,  1.]), array([ 0.,  1.,  2.]))
    >>> calcContigBlocksFromZ([0,1,1])
    (array([ 1.,  2.]), array([ 0.,  1.]))
    >>> calcContigBlocksFromZ([6,6,5])
    (array([ 2.,  1.]), array([ 0.,  2.]))

    Returns
    -------
    blockStarts : 1D array of size B
    blockSizes : 1D array of size B
    '''
    changePts = np.asarray(np.hstack([0,
                                      1 + np.flatnonzero(np.diff(Zvec)),
                                      len(Zvec)]), dtype=np.float64)
    assert len(changePts) >= 2
    chPtA = changePts[1:]
    chPtB = changePts[:-1]
    blockSizes = chPtA - chPtB
    blockStarts = np.asarray(changePts[:-1], dtype=np.float64)
    if returnStates:
        blockStates = Zvec[np.int32(blockStarts)].copy()
        if blockStates.size == 1 and blockStates.ndim == 0:
            blockStates = np.asarray([blockStates])
        return blockSizes, blockStarts, blockStates
    return blockSizes, blockStarts


def makeStateColorMap(nTrue=1, nExtra=0, nHighlight=0):
    '''
    Returns
    -------
    Cmap : ListedColormap object
    '''
    from matplotlib.colors import ListedColormap
    C = np.asarray([
        [166, 206, 227],
        [31, 120, 180],
        [178, 223, 138],
        [51, 160, 44],
        [251, 154, 153],
        [227, 26, 28],
        [254, 153, 41],
        [255, 127, 0],
        [202, 178, 214],
        [106, 61, 154],
        [223, 194, 125],
        [140, 81, 10],
        [128, 205, 193],
        [1, 102, 94],
        [241, 182, 218],
        [197, 27, 125],
    ], dtype=np.float64)
    C = np.vstack([C, 0.5 * C, 0.25 * C])
    if nTrue > C.shape[0]:
        raise ValueError('Cannot display more than %d true colors!' % (
            C.shape[0]))
    C = C[:nTrue] / 255.0
    shadeVals = np.linspace(0.2, 0.95, nExtra)
    for shadeID in range(nExtra):
        shadeOfRed = np.asarray([shadeVals[shadeID], 0, 0])
        C = np.vstack([C, shadeOfRed[np.newaxis, :]])

    highVals = np.linspace(0.3, 1.0, nHighlight)
    for highID in range(nHighlight):
        yellowColor = np.asarray([highVals[highID], highVals[highID], 0])
        C = np.vstack([C, yellowColor[np.newaxis, :]])

    return ListedColormap(C)


if __name__ == '__main__':
    import doctest
    doctest.run_docstring_examples(calcContigBlocksFromZ, globals())

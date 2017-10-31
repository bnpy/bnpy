from builtins import *
import numpy as np
import bnpy.viz
import os

from bnpy.viz.PlotComps import plotCompsFromSS
from bnpy.viz.PrintTopics import count2str

def cleanupDeleteSmallClusters(
        xSSslice, minNumAtomsToStay, xInitLPslice=None, pprintCountVec=None):
    ''' Remove all clusters with size less than specified amount.

    Returns
    -------
    xSSslice : SuffStatBag
        May have fewer components than K.
        Will not exactly represent data Dslice afterwards (if delete occurs).
    '''
    CountVec = xSSslice.getCountVec()
    badids = np.flatnonzero(CountVec < minNumAtomsToStay)
    massRemoved = np.sum(CountVec[badids])
    for k in reversed(badids):
        if xSSslice.K == 1:
            break
        xSSslice.removeComp(k)
        if xInitLPslice:
            xInitLPslice['DocTopicCount'] = np.delete(
                xInitLPslice['DocTopicCount'], k, axis=1)
    if pprintCountVec and badids.size > 0:
        pprintCountVec(xSSslice,
            cleanupMassRemoved=massRemoved,
            cleanupSizeThr=minNumAtomsToStay)
    if xInitLPslice:
        assert xInitLPslice['DocTopicCount'].shape[1] == xSSslice.K
    return xSSslice, xInitLPslice

def cleanupMergeClusters(
        xSSslice, curModel,
        xInitLPslice=None,
        obsSSkeys=None,
        vocabList=None,
        b_cleanupMaxNumMergeIters=3,
        b_cleanupMaxNumAcceptPerIter=1,
        b_mergeLam=None,
        b_debugOutputDir=None,
        pprintCountVec=None,
        **kwargs):
    ''' Merge all possible pairs of clusters that improve the Ldata objective.

    Returns
    -------
    xSSslice : SuffStatBag
        May have fewer components than K.
    '''
    xSSslice.removeELBOandMergeTerms()
    xSSslice.removeSelectionTerms()
    # Discard all fields unrelated to observation model
    reqFields = set()
    for key in obsSSkeys:
        reqFields.add(key)
    for key in list(xSSslice._Fields._FieldDims.keys()):
        if key not in reqFields:
            xSSslice.removeField(key)

    # For merges, we can crank up value of the topic-word prior hyperparameter,
    # to prioritize only care big differences in word counts across many terms
    tmpModel = curModel.copy()
    if b_mergeLam is not None:
        tmpModel.obsModel.Prior.lam[:] = b_mergeLam

    mergeID = 0
    for trial in range(b_cleanupMaxNumMergeIters):
        tmpModel.obsModel.update_global_params(xSSslice)
        GainLdata = tmpModel.obsModel.calcHardMergeGap_AllPairs(xSSslice)

        triuIDs = np.triu_indices(xSSslice.K, 1)
        posLocs = np.flatnonzero(GainLdata[triuIDs] > 0)
        if posLocs.size == 0:
            # No merges to accept. Stop!
            break

        # Rank the positive pairs from largest to smallest
        sortIDs = np.argsort(-1 * GainLdata[triuIDs][posLocs])
        posLocs = posLocs[sortIDs]

        usedUIDs = set()
        nAccept = 0
        uidpairsToAccept = list()
        origidsToAccept = list()
        for loc in posLocs:
            kA = triuIDs[0][loc]
            kB = triuIDs[1][loc]
            uidA = xSSslice.uids[triuIDs[0][loc]]
            uidB = xSSslice.uids[triuIDs[1][loc]]
            if uidA in usedUIDs or uidB in usedUIDs:
                continue
            usedUIDs.add(uidA)
            usedUIDs.add(uidB)
            uidpairsToAccept.append((uidA, uidB))
            origidsToAccept.append((kA, kB))
            nAccept += 1
            if nAccept >= b_cleanupMaxNumAcceptPerIter:
                break

        for posID, (uidA, uidB) in enumerate(uidpairsToAccept):
            mergeID += 1
            kA, kB = origidsToAccept[posID]
            xSSslice.mergeComps(uidA=uidA, uidB=uidB)

            if xInitLPslice:
                xInitLPslice['DocTopicCount'][:, kA] += \
                    xInitLPslice['DocTopicCount'][:, kB]
                xInitLPslice['DocTopicCount'][:, kB] = -1

            if b_debugOutputDir:
                savefilename = os.path.join(
                    b_debugOutputDir, 'MergeComps_%d.png' % (mergeID))
                # Show side-by-side topics
                bnpy.viz.PlotComps.plotCompsFromHModel(
                    tmpModel,
                    compListToPlot=[kA, kB],
                    vocabList=vocabList,
                    xlabels=[str(uidA), str(uidB)],
                    )
                bnpy.viz.PlotUtil.pylab.savefig(
                    savefilename, pad_inches=0)

        if len(uidpairsToAccept) > 0:
            pprintCountVec(xSSslice, uidpairsToAccept=uidpairsToAccept)

        if xInitLPslice:
            badIDs = np.flatnonzero(xInitLPslice['DocTopicCount'][0,:] < 0)
            for kk in reversed(badIDs):
                xInitLPslice['DocTopicCount'] = np.delete(
                    xInitLPslice['DocTopicCount'], kk, axis=1)

    if mergeID > 0 and b_debugOutputDir:
        tmpModel.obsModel.update_global_params(xSSslice)
        outpath = os.path.join(b_debugOutputDir, 'NewComps_AfterMerge.png')
        plotCompsFromSS(
            tmpModel, xSSslice, outpath,
            vocabList=vocabList,
            )

    if xInitLPslice:
        assert xInitLPslice['DocTopicCount'].min() > -0.000001
        assert xInitLPslice['DocTopicCount'].shape[1] == xSSslice.K
    return xSSslice, xInitLPslice

'''
    for loc in reversed(posLocs):
        kA = triuIDs[0][loc]
        kB = triuIDs[1][loc]
        if xSSslice.K > kB:
            mergeID += 1
            if b_debugOutputDir:
                savefilename = os.path.join(
                    b_debugOutputDir, 'MergeComps_%d.png' % (mergeID))
                # Show side-by-side topics
                bnpy.viz.PlotComps.plotCompsFromHModel(
                    tmpModel,
                    compListToPlot=[kA, kB],
                    vocabList=vocabList,
                    xlabels=[str(xSSslice.uids[kA]), str(xSSslice.uids[kB])],
                    )
                bnpy.viz.PlotUtil.pylab.savefig(
                    savefilename, pad_inches=0, bbox_inches='tight')
            xSSslice.mergeComps(kA, kB)
'''

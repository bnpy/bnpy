"""
Functions for collecting a target dataset for a delete move.

- addDataFromBatchToPlan
- getDataSubsetRelevantToPlan
"""

import numpy as np
import DeleteLogger


def addDataFromBatchToPlan(Plan, hmodel, Dchunk, LPchunk,
                           uIDs=None,
                           maxUID=-1,
                           batchID=0,
                           lapFrac=None,
                           isFirstBatch=0,
                           isLastBatch=0,
                           dtargetMaxSize=1000,
                           dtargetMinCount=0.01,
                           **kwargs):
    """ Add relevant data from provided chunk to the planned target set.

    Returns
    -------
    Plan : dict, same reference as provided, updated in-place.

    Post Conditions
    -------
    Plan dict is updated if current chunk has items to add to target set.
    Updated fields:
    * DTargetData
    * batchIDs

    Plan dict will be returned empty if:
    * Target set goes over the budget space of dtargetMaxSize
    * Target set has no items after the last batch.
    """
    assert uIDs is not None
    # Remember that recent seqcreate moves
    # can create more states in local params
    # than are currently available in the whole-dataset model,
    # because global step hasn't happened yet.
    assert len(uIDs) >= hmodel.allocModel.K
    assert len(uIDs) >= hmodel.obsModel.K

    if isFirstBatch:
        msg = '<<<<<<<<<<<<<<<<<<<< addDataFromBatchToPlan @ lap %6.2f' \
              % (np.ceil(lapFrac))
        DeleteLogger.log(msg)

    relData, relIDs = getDataSubsetRelevantToPlan(
        Dchunk, LPchunk, Plan,
        dtargetMinCount=dtargetMinCount)
    relSize = getSize(relData)
    if relSize < 1:
        msg = ' %6.3f | batch %3d | batch trgtSize 0 | agg trgtSize 0' \
              % (lapFrac, batchID)
        DeleteLogger.log(msg)

        if isLastBatch and not hasValidKey(Plan, 'DTargetData'):
            DeleteLogger.log("ABANDONED. No relevant items found.")
            return dict()
        return Plan

    # ----    Add all these docs to the Plan
    batchIDs = [batchID for n in range(relSize)]
    if hasValidKey(Plan, 'DTargetData'):
        Plan['DTargetData'].add_data(relData)
        Plan['batchIDs'].extend(batchIDs)
    else:
        Plan['DTargetData'] = relData
        Plan['batchIDs'] = batchIDs
        Plan['dataUnitIDs'] = relIDs

    curTargetSize = getSize(Plan['DTargetData'])
    if curTargetSize > dtargetMaxSize:
        for key in list(Plan.keys()):
            del Plan[key]
        msg = ' %6.3f | batch %3d | targetSize %d EXCEEDED BUDGET of %d' \
            % (lapFrac, batchID, curTargetSize, dtargetMaxSize)
        DeleteLogger.log(msg)
        DeleteLogger.log("ABANDONED.")
        return Plan

    if lapFrac is not None:
        msg = ' %6.3f | batch %3d | batch trgtSize %5d | agg trgtSize %5d' \
            % (lapFrac, batchID, relSize, curTargetSize)
        DeleteLogger.log(msg)

    # ----    Track stats specific to chosen subset
    targetLPchunk = hmodel.allocModel.selectSubsetLP(Dchunk, LPchunk, relIDs)
    targetSSchunk = hmodel.get_global_suff_stats(relData, targetLPchunk,
                                                 doPrecompEntropy=1)
    targetSSchunk.uIDs = uIDs.copy()

    # ----   targetSS tracks aggregate stats across batches
    if not hasValidKey(Plan, 'targetSS'):
        Kextra = 0
        Plan['targetSS'] = targetSSchunk.copy()
    else:
        Kextra = targetSSchunk.K - Plan['targetSS'].K
        if Kextra > 0:
            Plan['targetSS'].insertEmptyComps(Kextra)
        Plan['targetSS'] += targetSSchunk
        curUIDs = Plan['targetSS'].uIDs
        newUIDs = np.arange(maxUID - Kextra + 1, maxUID + 1)
        Plan['targetSS'].uIDs = np.hstack([curUIDs, newUIDs])

    # ----    targetSSByBatch tracks batch-specific stats
    if not hasValidKey(Plan, 'targetSSByBatch'):
        Plan['targetSSByBatch'] = dict()
    Plan['targetSSByBatch'][batchID] = targetSSchunk

    if np.allclose(lapFrac, np.ceil(lapFrac)):
        # Update batch-specific info
        # to account for any recent births
        for batchID in Plan['targetSSByBatch']:
            Kcur = Plan['targetSSByBatch'][batchID].K
            Kfinal = targetSSchunk.K
            Kextra = Kfinal - Kcur
            if Kextra > 0:
                curUIDs = Plan['targetSSByBatch'][batchID].uIDs
                newUIDs = np.arange(maxUID - Kextra + 1, maxUID + 1)
                newUIDs = np.hstack([curUIDs, newUIDs])

                del Plan['targetSSByBatch'][batchID].uIDs
                Plan['targetSSByBatch'][batchID].insertEmptyComps(Kextra)
                Plan['targetSSByBatch'][batchID].uIDs = newUIDs

    return Plan


def getDataSubsetRelevantToPlan(Dchunk, LPchunk, Plan,
                                dtargetMinCount=0.01):
    """ Get subset of provided DataObj containing units relevant to the Plan.

    Returns
    --------
    relData : None or bnpy.data.DataObj
    relIDs : list of integer ids of relevant units of provided Dchunk
    """
    if not hasValidKey(Plan, 'candidateIDs'):
        return None, []

    for dd, delCompID in enumerate(Plan['candidateIDs']):
        if 'DocTopicCount' in LPchunk:
            DocTopicCount = LPchunk['DocTopicCount']
            curkeepmask = DocTopicCount[:, delCompID] >= dtargetMinCount
        elif 'respPair' in LPchunk or 'TransCount' in LPchunk:
            curkeepmask = np.zeros(Dchunk.nDoc, dtype=np.int32)
            for n in range(Dchunk.nDoc):
                start = Dchunk.doc_range[n]
                stop = Dchunk.doc_range[n + 1]
                Usage_n = np.sum(LPchunk['resp'][start:stop, delCompID])
                curkeepmask[n] = Usage_n >= dtargetMinCount
        else:
            curkeepmask = LPchunk['resp'][:, delCompID] >= dtargetMinCount

        # Aggregate current mask with masks for all previous delCompID values
        if dd > 0:
            keepmask = np.logical_or(keepmask, curkeepmask)
        else:
            keepmask = curkeepmask

    relUnitIDs = np.flatnonzero(keepmask)
    if len(relUnitIDs) < 1:
        return None, relUnitIDs
    else:
        relData = Dchunk.select_subset_by_mask(relUnitIDs,
                                               doTrackFullSize=False)
        return relData, relUnitIDs


def hasValidKey(dict, key):
    """ Return True if key is in dict and not None, False otherwise.
    """
    return key in dict and dict[key] is not None


def getSize(Data):
    """ Return the integer size of the provided dataset.
    """
    if Data is None:
        return 0
    elif hasattr(Data, 'nDoc'):
        return Data.nDoc
    else:
        return Data.nObs

"""
Functions for planning a delete move.

- makePlanForEmptyComps
- makePlanForEligibleComps
- getEligibleCompInfo
- getEligibleCount
"""

import numpy as np
from bnpy.deletemove import DeleteLogger


def makePlanForEmptyComps(curSS, dtargetMinCount=0.01, **kwargs):
    """ Create a Plan dict for any empty states.

        Returns
        -------
        Plan : dict with either no fields, or two fields named
               * candidateIDs
               * candidateUIDs

        Any "empty" Plan dict indicates that no empty comps exist.
    """
    Nvec = curSS.getCountVec()
    emptyIDs = np.flatnonzero(Nvec < dtargetMinCount)
    if len(emptyIDs) == 0:
        return dict()
    Plan = dict(candidateIDs=emptyIDs.tolist(),
                candidateUIDs=curSS.uIDs[emptyIDs].tolist(),
                )
    return Plan


def makePlanForEligibleComps(SS, DRecordsByComp=None,
                             dtargetMaxSize=10,
                             deleteFailLimit=2,
                             lapFrac=-1,
                             **kwargs):
    ''' Create a Plan dict for any non-empty states eligible for a delete move.

    Really just a thin wrapper around getEligibleCompInfo,
    that does logging and verification of correctness.

    Returns
    -------
    Plan : dict with either no fields, or fields named
    * candidateIDs
    * candidateUIDs

    Any "empty" Plan dict indicates that no eligible comps exist.
    '''

    if lapFrac > -1:
        msg = '<<<<<<<<<<<<<<<<<<<< makePlanForEligibleComps @ lap %.2f' \
              % (np.ceil(lapFrac))
        DeleteLogger.log(msg)

    Plan = getEligibleCompInfo(SS, DRecordsByComp, dtargetMaxSize,
                               deleteFailLimit,
                               **kwargs)
    if 'candidateUIDs' in Plan:
        nEligibleBySize = len(Plan['eligible-by-size-UIDs'])
        nRemovedByFailLimit = len(Plan['eliminatedUIDs'])
        nFinalCandidates = len(Plan['candidateUIDs'])
    else:
        nEligibleBySize = 0

    DeleteLogger.log('%d/%d UIDs are eligible by size (1 <= size <= %d)' % (
        SS.K - (Plan['nEmpty'] + Plan['nTooBig']), SS.K, dtargetMaxSize))
    DeleteLogger.log('  skipped %d/%d UIDs that are empty (size < 1)' % (
        Plan['nEmpty'], SS.K))
    DeleteLogger.log('  skipped %d/%d UIDs that are too big (size > %d)' % (
        Plan['nTooBig'], SS.K, dtargetMaxSize))

    if nEligibleBySize == 0:
        DeleteLogger.log('  smallest non-empty UID has size: %d' % (
            Plan['minTooBigSize']))
        return dict()
    else:
        DeleteLogger.log('Eligible UIDs:')

        eUIDs = Plan['eligible-by-size-UIDs']
        sizeVec = [Plan['SizeMap'][x] for x in eUIDs]
        failVec = np.zeros_like(sizeVec)
        for ii in range(failVec.size):
            if 'nFail' in DRecordsByComp[eUIDs[ii]]:
                failVec[ii] = DRecordsByComp[eUIDs[ii]]['nFail']
            else:
                failVec[ii] = 0
        DeleteLogger.logPosVector(eUIDs, fmt='%5d', prefix='  UIDs:')
        DeleteLogger.logPosVector(sizeVec, fmt='%5d', prefix=' sizes:')
        DeleteLogger.logPosVector(failVec, fmt='%5d', prefix=' nFail:')

        DeleteLogger.log('Eligible UIDs eliminated by failure count:')
        if nRemovedByFailLimit > 0:
            DeleteLogger.logPosVector(Plan['eliminatedUIDs'], fmt='%5d')
        else:
            DeleteLogger.log('  None.')

        DeleteLogger.log('Num Tier1 = %d.  Num Tier2 = %d.'
                         % (Plan['nCandidateTier1'], Plan['nCandidateTier2']))
        DeleteLogger.log('Selected candidate comps: UIDs and sizes')
        if nFinalCandidates > 0:
            DeleteLogger.logPosVector(Plan['candidateUIDs'], fmt='%5d')
            DeleteLogger.logPosVector(Plan['candidateSizes'], fmt='%5d')
        else:
            DeleteLogger.log('  None. All disqualified.')
            return dict()

    return Plan


def getEligibleCompInfo(SS, DRecordsByComp=None,
                        dtargetMaxSize=10,
                        deleteFailLimit=2,
                        **kwargs):
    """ Get a dict of lists of component ids eligible for deletion.

    Returns
    -------
    Info : dict with either no fields, or fields named
        * candidateIDs
        * candidateUIDs

    Any "empty" Info dict indicates that no eligible comps exist.

    Post Condition
    -----------
    DRecordsByComp will have an entry for each candidate uID
    including updated fields:
    * 'count' : value of SS.getCountVec() corresponding to comp uID
    * 'nFail' : number of previous failed delete attempts on uID
    """
    assert hasattr(SS, 'uIDs')
    if DRecordsByComp is None:
        DRecordsByComp = dict()

    # ----    Measure size of each current state
    # CountVec refers to individual tokens/atoms
    CountVec = SS.getCountVec()

    # SizeVec refers to smallest-possible exchangeable units of data
    # e.g. documents in a topic-model, sequences for an HMM
    if SS.hasSelectionTerm('DocUsageCount'):
        SizeVec = SS.getSelectionTerm('DocUsageCount')
    else:
        raise NotImplementedError("DocUsageCount selection term required.")
        # SizeVec = 2 * CountVec  # conservative overestimate

    # ----    Find non-trivial states small enough to fit in target set
    mask_smallEnough = SizeVec <= dtargetMaxSize
    mask_tooBig = np.logical_not(mask_smallEnough)
    mask_nonTrivial = SizeVec >= 1
    eligibleIDs = np.flatnonzero(np.logical_and(
        mask_smallEnough, mask_nonTrivial))

    nEmpty = np.sum(1 - mask_nonTrivial)
    nTooBig = np.sum(1 - mask_smallEnough)

    if np.sum(mask_tooBig) > 0:
        minTooBigSize = SizeVec[mask_tooBig].min()
    else:
        minTooBigSize = -1

    eligibleUIDs = SS.uIDs[eligibleIDs]
    # ----    Return blank dict if no eligibles found
    if len(eligibleIDs) == 0:
        return dict(nEmpty=nEmpty,
                    nTooBig=nTooBig,
                    minTooBigSize=minTooBigSize)

    # sort these from smallest to largest usage
    sortIDs = np.argsort(SizeVec[eligibleIDs])
    eligibleIDs = eligibleIDs[sortIDs]
    eligibleUIDs = eligibleUIDs[sortIDs]

    # ----    Release potentially eligible UIDs from "failure" jail
    # If a state has changed mass significantly since last attempt,
    # we discard its past failures and make it eligible again.
    CountMap = dict()
    SizeMap = dict()
    for ii, uID in enumerate(eligibleUIDs):
        SizeMap[uID] = SizeVec[eligibleIDs[ii]]
        CountMap[uID] = CountVec[eligibleIDs[ii]]
    for uID in list(DRecordsByComp.keys()):
        if uID not in CountMap or 'count' not in DRecordsByComp[uID]:
            continue
        count = DRecordsByComp[uID]['count']
        percDiff = np.abs(CountMap[uID] - count) / (count + 1e-14)
        if percDiff > 0.15:
            del DRecordsByComp[uID]

    # ----    Update DRecordsByComp to track size of each eligible UID
    for uID in eligibleUIDs:
        if uID not in DRecordsByComp:
            DRecordsByComp[uID] = dict()
        if 'nFail' not in DRecordsByComp[uID]:
            DRecordsByComp[uID]['nFail'] = 0
        DRecordsByComp[uID]['count'] = CountMap[uID]

    # ----    Prioritize eligible comps by
    # -       * size (smaller preferred)
    # -       * previous failures (fewer preferred)
    tier1UIDs = list()
    tier2UIDs = list()
    eliminatedUIDs = list()

    for uID in eligibleUIDs:
        if DRecordsByComp[uID]['nFail'] == 0:
            tier1UIDs.append(uID)
        elif DRecordsByComp[uID]['nFail'] < deleteFailLimit:
            tier2UIDs.append(uID)
        else:
            # Any uID here is ineligible for a delete proposal.
            eliminatedUIDs.append(uID)

    # Select as many first tier as possible
    # until the target dataset budget is exceeded
    if hasattr(SS, 'nDoc'):
        totalSize = SS.nDoc
    else:
        totalSize = np.floor(CountVec.sum())
    canTakeEverything = dtargetMaxSize >= totalSize
    if len(tier1UIDs) > 0:
        tier1AggSize = np.cumsum([SizeMap[uID] for uID in tier1UIDs])

        if canTakeEverything:
            selectUIDs = tier1UIDs
            curTargetSize = dtargetMaxSize
        else:
            # maxLoc is an integer in {0, 1, ... |tier1UIDs|}
            # maxLoc equals m if we want everything in half-open interval 0:m
            # If we are looking for sizes <= dtargetMaxSize, need to add one
            # >>> searchsorted([3., 4., 5.], 3)
            # 0
            # >>> searchsorted([3., 4., 5.], 3+1)
            # 1
            maxLoc = np.searchsorted(tier1AggSize, dtargetMaxSize + 1)
            selectUIDs = tier1UIDs[:maxLoc]
            if maxLoc > 0:
                curTargetSize = tier1AggSize[maxLoc - 1]
            else:
                # We took no items from this tier
                curTargetSize = 0
    else:
        selectUIDs = []
        curTargetSize = 0
    selectUIDs = np.asarray(selectUIDs)

    # Fill remaining budget from second tier
    if canTakeEverything:
        selectUIDs = np.hstack([selectUIDs, tier2UIDs])
    elif curTargetSize < dtargetMaxSize:
        tier2AggSize = np.cumsum([SizeMap[x] for x in tier2UIDs])
        maxLoc = np.searchsorted(
            tier2AggSize,
            dtargetMaxSize +
            1 -
            curTargetSize)
        if maxLoc > 0:
            selectUIDs = np.hstack([selectUIDs, tier2UIDs[:maxLoc]])

    selectMassVec = [SizeMap[x] for x in selectUIDs]
    selectIDs = list()
    for uid in selectUIDs:
        jj = np.flatnonzero(uid == eligibleUIDs)[0]
        selectIDs.append(eligibleIDs[jj])

    Output = dict(CountMap=CountMap,
                  SizeMap=SizeMap,
                  minTooBigSize=minTooBigSize,
                  nEmpty=nEmpty,
                  nTooBig=nTooBig)
    Output['eligible-by-size-IDs'] = eligibleIDs
    Output['eligible-by-size-UIDs'] = eligibleUIDs
    Output['eliminatedUIDs'] = eliminatedUIDs
    Output['tier1UIDs'] = tier1UIDs
    Output['tier2UIDs'] = tier2UIDs
    Output['SizeMap'] = SizeMap
    Output['nCandidateTier1'] = len([x for x in selectUIDs if x in tier1UIDs])
    Output['nCandidateTier2'] = len([x for x in selectUIDs if x in tier2UIDs])
    Output['candidateIDs'] = selectIDs
    Output['candidateUIDs'] = selectUIDs.tolist()
    Output['candidateSizes'] = selectMassVec
    return Output


def getEligibleCount(SS, **kwargs):
    """ Get count of all current active comps eligible for deletion

        Returns
        -------
        count : int
    """
    Plan = getEligibleCompInfo(SS, **kwargs)
    if 'tier1UIDs' in Plan:
        nTotalEligible = len(Plan['tier1UIDs']) + len(Plan['tier2UIDs'])
    else:
        nTotalEligible = 0
    return nTotalEligible

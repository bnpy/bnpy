import numpy as np
from collections import defaultdict

import BLogger
from bnpy.viz.PrintTopics import vec2str
from bnpy.util import argsort_bigtosmall_stable, argsortBigToSmallByTiers

def selectCompsForBirthAtCurrentBatch(
        hmodel=None,
        SS=None,
        SSbatch=None,
        MoveRecordsByUID=dict(),
        MovePlans=dict(),
        lapFrac=0,
        batchID=0,
        batchPos=0,
        nBatch=1,
        isFirstBatch=False,
        doPrintLotsOfDetails=True,
        **BArgs):
    ''' Select specific comps to target with birth move at current batch.

    Returns
    -------
    MovePlans : dict with updated fields
    * b_targetUIDs : list of ints,
        Each uid in b_targetUIDs will be tried immediately, at current batch.

    MoveRecordsByUID : dict with updated fields
    * [uid]['byBatch'][batchID] : dict with fields
        proposalBatchSize
        proposalTotalSize
    '''
    # Extract num clusters in current model
    K = SS.K
    if K > 25:
        doPrintLotsOfDetails = False
    statusStr = ' lap %7.3f lapCeil %5d batchPos %3d/%d batchID %3d ' % (
        lapFrac, np.ceil(lapFrac), batchPos, nBatch, batchID)
    BLogger.pprint('PLAN at ' + statusStr)

    if BArgs['Kmax'] - SS.K <= 0:
        msg = "Cannot plan any more births." + \
            " Reached upper limit of %d existing comps (--Kmax)." % (
                BArgs['Kmax'])
        BLogger.pprint(msg)
        if 'b_targetUIDs' in MovePlans:
            del MovePlans['b_targetUIDs']
        MovePlans['b_statusMsg'] = msg
        BLogger.pprint('')
        return MovePlans

    if isFirstBatch:
        assert 'b_targetUIDs' not in MovePlans

    if isFirstBatch or 'b_firstbatchUIDs' not in MovePlans:
        MovePlans['b_firstbatchUIDs'] = SSbatch.uids.copy()
        MovePlans['b_CountVec_SeenThisLap'] = np.zeros(K)
    for k, uid in enumerate(MovePlans['b_firstbatchUIDs']):
        MovePlans['b_CountVec_SeenThisLap'][k] += SSbatch.getCountForUID(uid)

    # Short-circuit. Keep retained clusters.
    if lapFrac > 1.0 and BArgs['b_retainAcrossBatchesAfterFirstLap']:
        if not isFirstBatch:
            if 'b_targetUIDs' in MovePlans:
                msg = "%d UIDs retained from proposals earlier this lap." + \
                    " No new proposals at this batch."
                msg = msg % (len(MovePlans['b_targetUIDs']))
                BLogger.pprint(msg)
                if len(MovePlans['b_targetUIDs']) > 0:
                    BLogger.pprint(vec2str(MovePlans['b_targetUIDs']))
            else:
                BLogger.pprint(
                    'No UIDs targeted earlier in lap.' + \
                    ' No new proposals at this batch.')
            return MovePlans

    # Compute sizes for each cluster
    CountVec_b = np.maximum(SSbatch.getCountVec(), 1e-100)
    CountVec_all = np.maximum(SS.getCountVec(), 1e-100)
    atomstr = 'atoms'
    labelstr = 'count_b'

    uidsBusyWithOtherMoves = list()
    uidsTooSmall = list()
    uidsWithFailRecord = list()
    eligible_mask = np.zeros(K, dtype=np.bool8)
    for ii, uid in enumerate(SS.uids):
        if uid not in MoveRecordsByUID:
            MoveRecordsByUID[uid] = defaultdict(int)
        if not isinstance(MoveRecordsByUID[uid]['byBatch'], dict):
            MoveRecordsByUID[uid]['byBatch'] = \
                defaultdict(lambda: defaultdict(int))
        uidRec = MoveRecordsByUID[uid]
        uidRec_b = MoveRecordsByUID[uid]['byBatch'][batchID]

        uidstatusStr = "STATUS uid %5d %s N_b %9.3f N_ttl %9.3f" % (
            uid, statusStr,
            SSbatch.getCountForUID(uid), SS.getCountForUID(uid))
        # Continue to track UIDs that are pre-existing targets
        if 'b_targetUIDs' in MovePlans:
            if uid in MovePlans['b_targetUIDs']:
                BLogger.startUIDSpecificLog(uid)
                BLogger.pprint(uidstatusStr + " CHOSENAGAIN")
                BLogger.stopUIDSpecificLog(uid)
                continue
        # TODO REMOVE DEAD CODE
        if MoveRecordsByUID[uid]['b_tryAgainFutureLap'] > 0:
            msg = "Try targeting uid %d again." % (uid)
            BLogger.pprint(msg)
            del MoveRecordsByUID[uid]['b_tryAgainFutureLap']
            eligible_mask[ii] = 1
            continue

        # Discard uids which are active in another proposal.
        if 'd_targetUIDs' in MovePlans:
            if uid in MovePlans['d_targetUIDs']:
                uidsBusyWithOtherMoves.append(uid)
                BLogger.startUIDSpecificLog(uid)
                BLogger.pprint(uidstatusStr + " BUSY DELETE PROPOSAL")
                BLogger.stopUIDSpecificLog(uid)
                continue
        if 'd_absorbingUIDSet' in MovePlans:
            if uid in MovePlans['d_absorbingUIDSet']:
                uidsBusyWithOtherMoves.append(uid)
                BLogger.startUIDSpecificLog(uid)
                BLogger.pprint(uidstatusStr + " BUSY DELETE PROPOSAL")
                BLogger.stopUIDSpecificLog(uid)
                continue

        if 'm_targetUIDSet' in MovePlans:
            if uid in MovePlans['m_targetUIDSet']:
                uidsBusyWithOtherMoves.append(uid)
                BLogger.startUIDSpecificLog(uid)
                BLogger.pprint(uidstatusStr + " BUSY MERGE PROPOSAL")
                BLogger.stopUIDSpecificLog(uid)
                continue

        # Filter out uids without large presence in current batch
        bigEnough = CountVec_b[ii] >= BArgs['b_minNumAtomsForTargetComp']
        if not bigEnough:
            uidsTooSmall.append((uid, CountVec_b[ii]))
            BLogger.startUIDSpecificLog(uid)
            BLogger.pprint(uidstatusStr + " TOO SMALL %.2f < %.2f" % (
                CountVec_b[ii], BArgs['b_minNumAtomsForTargetComp']))
            BLogger.stopUIDSpecificLog(uid)
            continue

        eligibleSuffix = ''
        # Filter out uids we've failed on this particular batch before
        if uidRec_b['nFail'] > 0:
            prevBatchSize = uidRec_b['proposalBatchSize']
            prevTotalSize = uidRec_b['proposalTotalSize']

            curBatchSize = SSbatch.getCountForUID(uid)
            sizePercDiff = np.abs(curBatchSize - prevBatchSize) / (
                curBatchSize + 1e-100)
            sizeChangedEnoughToReactivate = sizePercDiff > \
                BArgs['b_minPercChangeInNumAtomsToReactivate']

            curTotalSize = SS.getCountForUID(uid)
            totalPercDiff = np.abs(curTotalSize - prevTotalSize) / (
                curTotalSize + 1e-100)
            totalsizeChangedEnoughToReactivate = totalPercDiff > \
                BArgs['b_minPercChangeInNumAtomsToReactivate']

            if sizeChangedEnoughToReactivate:
                eligibleSuffix = \
                    "REACTIVATE BY BATCH SIZE." + \
                    "\n Batch size percDiff %.2f > %.2f" % (
                        sizePercDiff,
                        BArgs['b_minPercChangeInNumAtomsToReactivate']) \
                    + "\n prevBatchSize %9.2f \n curBatchSize %9.2f" % (
                        prevBatchSize, curBatchSize)
                uidRec_b['nFail'] = 0 # Reactivated
            elif totalsizeChangedEnoughToReactivate:
                eligibleSuffix = \
                    "REACTIVATED BY TOTAL SIZE" + \
                    "\n Total size percDiff %.2f > %.2f" % (
                        totalPercDiff,
                        BArgs['b_minPercChangeInNumAtomsToReactivate']) \
                    + "\n prevTotalSize %9.1f \n curTotalSize %9.1f" % (
                        prevTotalSize, curTotalSize)
                uidRec_b['nFail'] = 0 # Reactivated
            else:
                uidsWithFailRecord.append(uid)
                BLogger.startUIDSpecificLog(uid)
                BLogger.pprint(
                    uidstatusStr + " DISQUALIFIED FOR PAST FAILURE")
                BLogger.stopUIDSpecificLog(uid)
                continue
        # If we've made it here, the uid is eligible.
        eligible_mask[ii] = 1
        BLogger.startUIDSpecificLog(uid)
        BLogger.pprint(uidstatusStr + " ELIGIBLE " + eligibleSuffix)
        BLogger.stopUIDSpecificLog(uid)


    # Notify about uids retained
    if 'b_targetUIDs' not in MovePlans:
        MovePlans['b_targetUIDs'] = list()
    msg = "%d/%d UIDs retained from preexisting proposals." % (
        len(MovePlans['b_targetUIDs']), K)
    BLogger.pprint(msg)

    # Log info about busy disqualifications
    nDQ_toobusy = len(uidsBusyWithOtherMoves)
    nDQ_pastfail = len(uidsWithFailRecord)
    msg = "%d/%d UIDs too busy with other moves (merge/delete)." % (
        nDQ_toobusy, K)
    BLogger.pprint(msg)
    # Log info about toosmall disqualification
    nDQ_toosmall = len(uidsTooSmall)
    msg = "%d/%d UIDs too small (too few %s in current batch)." + \
        " Required size >= %d (--b_minNumAtomsForTargetComp)"
    msg = msg % (nDQ_toosmall, K, atomstr,
        BArgs['b_minNumAtomsForTargetComp'])
    BLogger.pprint(msg, 'debug')
    if nDQ_toosmall > 0 and doPrintLotsOfDetails:
        lineUID = vec2str([u[0] for u in uidsTooSmall])
        lineSize = vec2str([u[1] for u in uidsTooSmall])
        BLogger.pprint([lineUID, lineSize], 
            prefix=['%7s' % 'uids',
                    '%7s' % labelstr],
            )
    # Notify about past failure disqualifications to the log
    BLogger.pprint(
        '%d/%d UIDs disqualified for past failures.' % (
            nDQ_pastfail, K),
        'debug')
    if nDQ_pastfail > 0 and doPrintLotsOfDetails:
        lineUID = vec2str(uidsWithFailRecord)
        BLogger.pprint(lineUID)
    # Store nDQ counts for reporting.
    MovePlans['b_nDQ_toosmall'] = nDQ_toosmall
    MovePlans['b_nDQ_toobusy'] = nDQ_toobusy
    MovePlans['b_nDQ_pastfail'] = nDQ_pastfail
    # Finalize list of eligible UIDs
    eligibleUIDs = SS.uids[eligible_mask]
    BLogger.pprint('%d/%d UIDs eligible for new proposal' % (
        len(eligibleUIDs), K))
    # EXIT if nothing eligible.
    if len(eligibleUIDs) == 0:
        BLogger.pprint('')
        assert 'b_targetUIDs' in MovePlans
        return MovePlans

    # Record all uids that are eligible!
    # And make vector of how recently they have failed in other attempts
    FailVec = np.inf * np.ones(K)
    for uid in eligibleUIDs:
        uidRec['b_latestEligibleLap'] = lapFrac
        k = SS.uid2k(uid)
        FailVec[k] = MoveRecordsByUID[uid]['b_nFailRecent']

    if doPrintLotsOfDetails:
        lineUID = vec2str(eligibleUIDs)
        lineSize = vec2str(CountVec_all[eligible_mask])
        lineBatchSize = vec2str(CountVec_b[eligible_mask])
        lineFail = vec2str(FailVec[eligible_mask])
        BLogger.pprint([lineUID, lineSize, lineBatchSize, lineFail],
                prefix=[
                    '%7s' % 'uids',
                    '%7s' % 'cnt_ttl',
                    '%7s' % 'cnt_b',
                    '%7s' % 'nFail',
                    ],
                )

    # Figure out how many new states we can target this round.
    # Prioritize the top comps as ranked by the desired score
    # until we max out the budget of Kmax total comps.
    maxnewK = BArgs['Kmax'] - SS.K
    totalnewK_perEligibleComp = np.minimum(
        np.ceil(CountVec_b[eligible_mask]),
        np.minimum(BArgs['b_Kfresh'], maxnewK))
    # TODO: Worry about retained ids with maxnewK
    sortorder = argsortBigToSmallByTiers(
        -1 * FailVec[eligible_mask], CountVec_b[eligible_mask])
    sortedCumulNewK = np.cumsum(totalnewK_perEligibleComp[sortorder])
    nToKeep = np.searchsorted(sortedCumulNewK, maxnewK + 0.0042)
    if nToKeep == 0:
        nToKeep = 1        
    keepEligibleIDs = sortorder[:nToKeep]
    newK = np.minimum(sortedCumulNewK[nToKeep-1], maxnewK)
    chosenUIDs = [eligibleUIDs[s] for s in keepEligibleIDs]

    if nToKeep < len(chosenUIDs):
        BLogger.pprint(
            'Selected %d/%d eligible UIDs to do proposals.' % (
                nToKeep, len(chosenUIDs)) + \
            '\n Could create up to %d new clusters, %d total clusters.' % (
                newK, newK + SS.K) + \
            '\n Total budget allows at most %d clusters (--Kmax).' % (
                BArgs['Kmax']),
            )
    BLogger.pprint('%d/%d UIDs chosen for new proposals (rankby: cnt_b)' % (
        len(chosenUIDs), len(eligibleUIDs)))
    if doPrintLotsOfDetails:
        lineUID = vec2str(chosenUIDs)
        lineSize = vec2str(CountVec_all[eligible_mask][keepEligibleIDs])
        lineBatchSize = vec2str(CountVec_b[eligible_mask][keepEligibleIDs])
        lineFail = vec2str(FailVec[eligible_mask][keepEligibleIDs])
        BLogger.pprint([lineUID, lineSize, lineBatchSize, lineFail],
            prefix=[
                '%7s' % 'uids',
                '%7s' % 'cnt_ttl',
                '%7s' % 'cnt_b',
                '%7s' % 'fail',
                ],
            )

    for uid in chosenUIDs:
        uidRec = MoveRecordsByUID[uid]
        uidRec['b_proposalBatchID'] = batchID
        uidRec_b = MoveRecordsByUID[uid]['byBatch'][batchID]
        uidRec_b['proposalBatchSize'] = SSbatch.getCountForUID(uid)
        uidRec_b['proposalTotalSize'] = SSbatch.getCountForUID(uid)

    # Aggregate all uids
    MovePlans['b_newlyChosenTargetUIDs'] = chosenUIDs
    MovePlans['b_preExistingTargetUIDs'] = \
        [u for u in MovePlans['b_targetUIDs']]
    MovePlans['b_targetUIDs'].extend(chosenUIDs)

    BLogger.pprint('')
    return MovePlans


def selectShortListForBirthAtLapStart(
        hmodel, SS,
        MoveRecordsByUID=dict(),
        MovePlans=dict(),
        lapFrac=0,
        b_minNumAtomsForTargetComp=2,
        **BArgs):
    ''' Select list of comps to possibly target with birth during next lap.

    Shortlist uids are guaranteed to never be involved in a merge/delete.
    They are kept aside especially for a birth move, at least in this lap.
    
    Returns
    -------
    MovePlans : dict with updated fields
    * b_shortlistUIDs : list of ints,
        Each uid in b_shortlistUIDs could be a promising birth target.
        None of these should be touched by deletes or merges in this lap.
    '''
    MovePlans['b_shortlistUIDs'] = list()
    MovePlans['b_nDQ_toosmall'] = 0
    MovePlans['b_nDQ_pastfail'] = 0
    MovePlans['b_nDQ_toobusy'] = 0
    MovePlans['b_roomToGrow'] = 0
    MovePlans['b_maxLenShortlist'] = 0
    if not canBirthHappenAtLap(lapFrac, **BArgs):
        BLogger.pprint('')
        return MovePlans

    K = hmodel.obsModel.K
    KroomToGrow = BArgs['Kmax'] - K
    MovePlans['b_roomToGrow'] = KroomToGrow
    # Each birth adds at least 2 comps.
    # If we have 10 slots left, we can do at most 5 births
    maxLenShortlist = KroomToGrow / 2
    MovePlans['b_maxLenShortlist'] = maxLenShortlist

    # EXIT: early, if no room to grow.
    if KroomToGrow <= 1:
        BLogger.pprint(
            "Cannot shortlist any comps for birth." + \
            " Adding 2 more comps to K=%d exceeds limit of %d (--Kmax)." % (
                K, BArgs['Kmax'])
            )
        BLogger.pprint('')
        return MovePlans
    # Log reasons for shortlist length
    if maxLenShortlist < K:
        msg = " Limiting shortlist to %d possible births this lap." % (
            maxLenShortlist)
        msg += " Any more would cause current K=%d to exceed Kmax=%d" % (
            K, BArgs['Kmax'])
        BLogger.pprint(msg)
    # Handle initialization case: SS is None
    # Must just select all possible comps
    if SS is None:
        shortlistUIDs = np.arange(K).tolist()
        shortlistUIDs = shortlistUIDs[:maxLenShortlist]
        MovePlans['b_shortlistUIDs'] = shortlistUIDs
        BLogger.pprint(
            "No SS provided. Shortlist contains %d possible comps" % (
                len(shortlistUIDs)))
        BLogger.pprint('')
        return MovePlans
    assert SS.K == K

    CountVec = SS.getCountVec()
    eligible_mask = np.zeros(K, dtype=np.bool8)
    nTooSmall = 0
    nPastFail = 0
    for k, uid in enumerate(SS.uids):
        if uid not in MoveRecordsByUID:
            MoveRecordsByUID[uid] = defaultdict(int)
        tooSmall = CountVec[k] <= b_minNumAtomsForTargetComp
        hasFailRecord = MoveRecordsByUID[uid]['b_nFailRecent'] > 0        
        if MoveRecordsByUID[uid]['b_tryAgainFutureLap'] > 0:
            eligible_mask[k] = 1
            MovePlans['b_shortlistUIDs'].append(uid)
        elif (not tooSmall) and (not hasFailRecord):
            eligible_mask[k] = 1
            MovePlans['b_shortlistUIDs'].append(uid)
        elif tooSmall:
            nTooSmall += 1
        else:
            assert hasFailRecord
            nPastFail += 1
    assert len(MovePlans['b_shortlistUIDs']) == np.sum(eligible_mask)
    # Rank the shortlist by size
    if maxLenShortlist < len(MovePlans['b_shortlistUIDs']):
        sortIDs = argsort_bigtosmall_stable(CountVec[eligible_mask])
        sortIDs = sortIDs[:maxLenShortlist]
        MovePlans['b_shortlistUIDs'] = [
            MovePlans['b_shortlistUIDs'][s] for s in sortIDs]
        shortlistCountVec = CountVec[eligible_mask][sortIDs]
    else:
        shortlistCountVec = CountVec[eligible_mask]

    MovePlans['b_nDQ_toosmall'] = nTooSmall
    MovePlans['b_nDQ_pastfail'] = nPastFail
    nShortList = len(MovePlans['b_shortlistUIDs'])
    assert nShortList <= maxLenShortlist
    BLogger.pprint(
        "%d/%d uids selected for short list." % (nShortList, K))
    if nShortList > 0:
        lineUID = vec2str(MovePlans['b_shortlistUIDs'])
        lineSize = vec2str(shortlistCountVec)
        BLogger.pprint([lineUID, lineSize], 
            prefix=['%7s' % 'uids',
                    '%7s' % 'size'],
            )
    BLogger.pprint('')
    return MovePlans

def canBirthHappenAtLap(lapFrac, b_startLap=-1, b_stopLap=-1, **kwargs):
    ''' Make binary yes/no decision if birth move can happen at provided lap.

    Returns
    -------
    answer : boolean
        True only if lapFrac >= b_startLap and lapFrac < stopLap

    Examples
    --------
    >>> canBirthHappenAtLap(0.1, b_startLap=1, b_stopLap=2)
    True
    >>> canBirthHappenAtLap(1.0, b_startLap=1, b_stopLap=2)
    True
    >>> canBirthHappenAtLap(1.1, b_startLap=1, b_stopLap=2)
    False
    >>> canBirthHappenAtLap(2.0, b_startLap=1, b_stopLap=2)
    False
    >>> canBirthHappenAtLap(10.5, b_startLap=1, b_stopLap=2)
    False
    >>> canBirthHappenAtLap(10.5, b_startLap=1, b_stopLap=11)
    False
    >>> canBirthHappenAtLap(10.5, b_startLap=1, b_stopLap=12)
    True
    '''
    if b_startLap < 0:
        return False
    elif b_startLap >= 0 and np.ceil(lapFrac) < b_startLap:
        return False 
    elif b_stopLap >= 0 and np.ceil(lapFrac) >= b_stopLap:
        return False
    else:
        return True

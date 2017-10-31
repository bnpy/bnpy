from builtins import *
import numpy as np
from collections import defaultdict

from . import DLogger
import bnpy.birthmove.BPlanner as BPlanner
from bnpy.viz.PrintTopics import count2str, vec2str

def selectCandidateDeleteComps(
        hmodel, SS,
        MoveRecordsByUID=dict(),
        MovePlans=dict(),
        lapFrac=0,
        **DArgs):
    ''' Select specific comps to target with delete move.

    Returns
    -------
    MovePlans : dict, with fields
    * d_targetUIDs : list of ints
    * d_absorbingUIDSet : set of ints, all uids that can absorb target mass
    OR
    * failMsg : string explaining why building list of eligible UIDs failed
    '''
    DLogger.pprint("PLANNING delete at lap %.2f" % (lapFrac))
    K = SS.K

    availableUIDs = set(SS.uids)
    if len(availableUIDs) < 2:
        DLogger.pprint(
            "Delete proposal requires at least 2 available UIDs.\n" + \
            "   Need 1 uid to target, and at least 1 to absorb." + \
            "   Only have %d total uids in the model." % (len(availableUIDs)))
        failMsg = "Ineligible. Did not find >= 2 UIDs in entire model."
        return dict(failMsg=failMsg)

    uidsBusyWithOtherMoves = set()
    '''
    if 'm_UIDPairs' in MovePlans:
        for (uidA, uidB) in MovePlans['m_UIDPairs']:
            availableUIDs.discard(uidA)
            availableUIDs.discard(uidB)
            uidsBusyWithOtherMoves.add(uidA)
            uidsBusyWithOtherMoves.add(uidB)
    if 'b_shortlistUIDs' in MovePlans:
        for uid in MovePlans['b_shortlistUIDs']:
            availableUIDs.discard(uid)
            uidsBusyWithOtherMoves.add(uid)

    if len(availableUIDs) < 2:
        DLogger.pprint("Delete requires at least 2 UIDs" + \
            " not occupied by merge or birth.\n" + \
            "   Need 1 uid to target, and at least 1 to absorb.\n" + \
            "   Only have %d total uids eligible." % (len(availableUIDs)))
        failMsg = "Ineligible. Too many uids occupied by merge or shortlisted for birth."
        return dict(failMsg=failMsg)
    '''

    # Compute score for each eligible state
    countVec = np.maximum(SS.getCountVec(), 1e-100)
    eligibleUIDs = list()
    tooBigUIDs = list()
    failRecordUIDs = list()
    nFailRecord = 0
    nReactivated = 0
    for uid in availableUIDs:
        k = SS.uid2k(uid)
        size = countVec[k]
        if uid not in MoveRecordsByUID:
            MoveRecordsByUID[uid] = defaultdict(int)

        # Skip ahead if this cluster is too big
        if size > DArgs['d_maxNumAtomsForTargetComp']:
            tooBigUIDs.append(uid)
            continue
        # Avoid comps we've failed deleting in the past
        # unless they have changed by a reasonable amount
        # or enough laps have passed to try again
        lapsSinceLastTry = lapFrac - MoveRecordsByUID[uid]['d_latestLap']
        nFailRecent_Delete = MoveRecordsByUID[uid]['d_nFailRecent'] > 0
        oldsize = MoveRecordsByUID[uid]['d_latestCount']
        if oldsize > 0 and nFailRecent_Delete > 0:
            nFailRecord += 1
            sizePercDiff = np.abs(size - oldsize)/(1e-100 + np.abs(oldsize))
            if sizePercDiff > DArgs['d_minPercChangeInNumAtomsToReactivate']:
                nReactivated += 1
            elif DArgs['d_nLapToReactivate'] > 0 \
                    and lapsSinceLastTry > DArgs['d_nLapToReactivate']:
                nReactivated += 1
            else:
                failRecordUIDs.append(uid)
                continue
        # If we make it here, the uid is eligible
        eligibleUIDs.append(uid)

    # Log which uids were marked has high potential births
    msg = "%d/%d UIDs busy with other moves (birth/merge)" % (
       len(uidsBusyWithOtherMoves), K)
    DLogger.pprint(msg)
    if len(uidsBusyWithOtherMoves) > 0:
        DLogger.pprint(
            '  ' + vec2str(uidsBusyWithOtherMoves), 'debug')

    msg = "%d/%d UIDs too large [--d_maxNumAtomsForTargetComp %.2f]" % (
            len(tooBigUIDs), K, DArgs['d_maxNumAtomsForTargetComp'])
    DLogger.pprint(msg)
    if len(tooBigUIDs) > 0:
        DLogger.pprint(
            '  ' + vec2str(tooBigUIDs), 'debug')

    # Log which uids were marked has having a record.
    msg = '%d/%d UIDs un-deleteable for past failures. %d reactivated.' % (
        len(failRecordUIDs), K, nReactivated)
    DLogger.pprint(msg)
    if len(failRecordUIDs) > 0:
        DLogger.pprint(
            '  ' + vec2str(failRecordUIDs), 'debug')
    # Log all remaining eligible uids
    msg = '%d/%d UIDs eligible for targeted delete proposal' % (
        len(eligibleUIDs), K)
    DLogger.pprint(msg)
    if len(eligibleUIDs) == 0:
        failMsg = ("Empty plan. 0 UIDs eligible as delete target." + \
            " %d too busy with other moves." + \
            " %d too big." + \
            " %d have past failures.") % (
                len(uidsBusyWithOtherMoves),
                len(tooBigUIDs),
                len(failRecordUIDs))
        return dict(failMsg=failMsg)

    # Log count statistics for each uid
    eligibleCountVec = [countVec[SS.uid2k(u)] for u in eligibleUIDs]
    DLogger.pprint(
        ' uid   ' + vec2str(eligibleUIDs), 'debug')
    DLogger.pprint(
        ' count ' + vec2str(eligibleCountVec), 'debug')

    # Select the single state to target
    # by taking the one with highest score
    #Scores = np.asarray([x for x in ScoreByEligibleUID.values()])
    #targetUID = eligibleUIDs[np.argmax(eligibleCountVec)]
    #MovePlans['d_targetUIDs'] = [targetUID]

    targetUID = eligibleUIDs[np.argmax(eligibleCountVec)]
    MovePlans['d_targetUIDs'] = [targetUID]

    # Determine all comps eligible to receive its transfer mass
    absorbUIDset = set(eligibleUIDs)
    absorbUIDset.discard(targetUID)
    absorbUIDset.update(tooBigUIDs)
    absorbUIDset.update(failRecordUIDs)
    MovePlans['d_absorbingUIDSet'] = absorbUIDset

    DLogger.pprint('Selecting one single state to target.')
    DLogger.pprint('targetUID ' + str(targetUID))
    DLogger.pprint('absorbingUIDs: ' + vec2str(absorbUIDset))
    return MovePlans

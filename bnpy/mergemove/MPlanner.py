'''
MPlanner.py

Functions for deciding which merge pairs to track.
'''
from builtins import *
import numpy as np
import sys
import os
import itertools
from . import MLogger
from collections import defaultdict
from bnpy.viz.PrintTopics import vec2str, count2str


def selectCandidateMergePairs(hmodel, SS,
        MovePlans=dict(),
        MoveRecordsByUID=dict(),
        lapFrac=None,
        m_maxNumPairsContainingComp=3,
        m_minPercChangeInNumAtomsToReactivate=0.01,
        m_nLapToReactivate=10,
        m_pair_ranking_procedure='total_size',
        m_pair_ranking_direction='descending',
        m_pair_ranking_do_exclude_by_thr=0,
        m_pair_ranking_exclusion_thr=-0.000001,
        **kwargs):
    ''' Select candidate pairs to consider for merge move.

    Returns
    -------
    Info : dict, with fields
        * m_UIDPairs : list of tuples, each defining a pair of uids
        * m_targetUIDSet : set of all uids involved in a proposed merge pair
    '''
    MLogger.pprint(
        "PLANNING merges at lap %.2f. K=%d" % (lapFrac, SS.K),
        'debug')

    # Mark any targetUIDs used in births as off-limits for merges
    uidUsageCount = defaultdict(int)
    if 'b_shortlistUIDs' in MovePlans:
        for uid in MovePlans['b_shortlistUIDs']:
            uidUsageCount[uid] = 10 * m_maxNumPairsContainingComp
    nDisqualified = len(list(uidUsageCount.keys()))
    MLogger.pprint(
        "   %d/%d UIDs ineligible because on shortlist for births. " % (
            nDisqualified, SS.K),
        'debug')
    if nDisqualified > 0:
        MLogger.pprint(
            "   Ineligible UIDs:" + \
                vec2str(list(uidUsageCount.keys())),
            'debug')

    uid2k = dict()
    uid2count = dict()
    for uid in SS.uids:
        uid2k[uid] = SS.uid2k(uid)
        uid2count[uid] = SS.getCountForUID(uid)

    EligibleUIDPairs = list()
    EligibleAIDPairs = list()
    nPairTotal = 0
    nPairDQ = 0
    nPairBusy = 0
    for kA, uidA in enumerate(SS.uids):
        for b, uidB in enumerate(SS.uids[kA+1:]):
            kB = kA + b + 1
            assert kA < kB
            nPairTotal += 1
            if uidUsageCount[uidA] > 0 or uidUsageCount[uidB] > 0:
                nPairBusy += 1
                continue
            if uidA < uidB:
                uidTuple = (uidA, uidB)
            else:
                uidTuple = (uidB, uidA)
            aidTuple = (kA, kB)

            if uidTuple not in MoveRecordsByUID:
                EligibleUIDPairs.append(uidTuple)
                EligibleAIDPairs.append(aidTuple)
            else:
                pairRecord = MoveRecordsByUID[uidTuple]
                assert pairRecord['m_nFailRecent'] >= 1
                latestMinCount = pairRecord['m_latestMinCount']
                newMinCount = np.minimum(uid2count[uidA], uid2count[uidB])
                percDiff = np.abs(latestMinCount - newMinCount) / \
                    latestMinCount
                if (lapFrac - pairRecord['m_latestLap']) >= m_nLapToReactivate:
                    EligibleUIDPairs.append(uidTuple)
                    EligibleAIDPairs.append(aidTuple)
                    del MoveRecordsByUID[uidTuple]
                elif percDiff >= m_minPercChangeInNumAtomsToReactivate:
                    EligibleUIDPairs.append(uidTuple)
                    EligibleAIDPairs.append(aidTuple)
                    del MoveRecordsByUID[uidTuple]
                else:
                    nPairDQ += 1
    MLogger.pprint(
        "   %d/%d pairs eligible. %d disqualified by past failures." % (
            len(EligibleAIDPairs), nPairTotal, nPairDQ),
        'debug')
    MLogger.pprint(
        "   Prioritizing elible pairs via ranking procedure: %s" % (
            m_pair_ranking_procedure),
        'debug')
    if m_pair_ranking_procedure == 'random':
        A = len(EligibleAIDPairs)
        prng = np.random.RandomState(lapFrac)
        rank_scores_per_pair = prng.permutation(np.arange(A))
    elif m_pair_ranking_procedure == 'total_size':
        A = len(EligibleAIDPairs)
        rank_scores_per_pair = np.asarray(
            [SS.getCountForUID(uidA) + SS.getCountForUID(uidB)
                for (uidA, uidB) in EligibleUIDPairs])
    elif m_pair_ranking_procedure.count('elbo'):
        # Compute Ldata gain for each possible pair of comps
        rank_scores_per_pair = hmodel.obsModel.calcHardMergeGap_SpecificPairs(
            SS, EligibleAIDPairs)
        if hasattr(hmodel.allocModel, 'calcHardMergeGap_SpecificPairs'):
            rank_scores_per_pair = \
                rank_scores_per_pair + hmodel.allocModel.calcHardMergeGap_SpecificPairs(
                    SS, EligibleAIDPairs)
        rank_scores_per_pair /= hmodel.obsModel.getDatasetScale(SS)
    else:
        raise ValueError(
            "Unrecognised --m_pair_ranking_procedure: %s" %
                m_pair_ranking_procedure)

    # Find pairs with positive gains
    if m_pair_ranking_direction == 'ascending':
        if m_pair_ranking_do_exclude_by_thr:
            MLogger.pprint(
                "Keeping only uid pairs with score < %.3e" % (
                    m_pair_ranking_exclusion_thr),
                'debug')
            keep_pair_ids = np.flatnonzero(rank_scores_per_pair < m_pair_ranking_exclusion_thr)
            ranked_pair_locs = keep_pair_ids[
                 np.argsort(rank_scores_per_pair[keep_pair_ids])]
        else:
            ranked_pair_locs = np.argsort(rank_scores_per_pair)
    else:
        if m_pair_ranking_do_exclude_by_thr:
            MLogger.pprint(
                "Keeping only uid pairs with score > %.3e" % (
                    m_pair_ranking_exclusion_thr),
                'debug')
            keep_pair_ids = np.flatnonzero(rank_scores_per_pair >m_pair_ranking_exclusion_thr)
            ranked_pair_locs = keep_pair_ids[
                 np.argsort(-1 * rank_scores_per_pair[keep_pair_ids])]
        else:
            ranked_pair_locs = np.argsort(-1 * rank_scores_per_pair)

    nKeep = 0
    mUIDPairs = list()
    mAIDPairs = list()
    mGainVals = list()
    for loc in ranked_pair_locs:
        uidA, uidB = EligibleUIDPairs[loc]
        kA, kB = EligibleAIDPairs[loc]
        if uidUsageCount[uidA] >= m_maxNumPairsContainingComp or \
                uidUsageCount[uidB] >= m_maxNumPairsContainingComp:
            continue
        uidUsageCount[uidA] += 1
        uidUsageCount[uidB] += 1

        mAIDPairs.append((kA, kB))
        mUIDPairs.append((uidA, uidB))
        mGainVals.append(rank_scores_per_pair[loc])
        if nKeep == 0:
            MLogger.pprint("Chosen uid pairs:", 'debug')
        MLogger.pprint(
            "%4d, %4d : pair_score %.3e, size %s %s" % (
                uidA, uidB,
                rank_scores_per_pair[loc],
                count2str(uid2count[uidA]),
                count2str(uid2count[uidB]),
                ),
            'debug')
        nKeep += 1
    Info = dict()
    Info['m_UIDPairs'] = mUIDPairs
    Info['m_GainVals'] = mGainVals
    Info['mPairIDs'] = mAIDPairs
    targetUIDs = set()
    for uidA, uidB in mUIDPairs:
        targetUIDs.add(uidA)
        targetUIDs.add(uidB)
        if 'b_shortlistUIDs' in MovePlans:
            for uid in MovePlans['b_shortlistUIDs']:
                assert uid != uidA
                assert uid != uidB
    Info['m_targetUIDSet'] = targetUIDs
    return Info

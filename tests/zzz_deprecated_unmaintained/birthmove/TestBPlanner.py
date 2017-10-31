import numpy as np
import os
import unittest
from collections import defaultdict

from bnpy.suffstats import SuffStatBag
from bnpy.birthmove.BLogger import configure
from bnpy.birthmove.BPlanner import selectCompsForBirthAtCurrentBatch

configure('/tmp/', doSaveToDisk=0, doWriteStdOut=1, stdoutLevel=0)

BArgs = dict(
	Kmax=300,
	b_retainAcrossBatchesAfterFirstLap=True,
	b_minNumAtomsForTargetComp=1.0,
	b_Kfresh=2,
	doPrintLotsOfDetails=1)

def test_BPlanner_makePlanAtBatch_noPrevFailures(K=10):
	SS = SuffStatBag(K=K)
	SS.setField('N',  np.arange(K), dims='K')
	SSbatch = SS.copy()

	for b_minNumAtomsForTargetComp in [2, 5, K]:
		BArgs['b_minNumAtomsForTargetComp'] = b_minNumAtomsForTargetComp
		MovePlans = selectCompsForBirthAtCurrentBatch(
			SS=SS, SSbatch=SSbatch, MovePlans=dict(), **BArgs)
		nChosen = len(MovePlans['b_targetUIDs'])
		assert nChosen == np.sum(SS.N >= b_minNumAtomsForTargetComp)

def test_BPlanner_makePlanAtBatch_someDisqualifiedForPrevFailures(K=10):
	SS = SuffStatBag(K=K)
	SS.setField('N',  np.arange(K), dims='K')
	SSbatch = SS.copy()

	# Do the same test, while eliminating some uids
	MoveRecordsByUID = defaultdict(lambda: defaultdict(int))
	for uid in [0, 6, 9]:
		MoveRecordsByUID[uid]['b_nFail'] = 1
		MoveRecordsByUID[uid]['b_nFailRecent'] = 1
		MoveRecordsByUID[uid]['b_batchIDsWhoseProposalFailed'] = set([0])

	for b_minNumAtomsForTargetComp in [2, 5, K]:
		BArgs['b_minNumAtomsForTargetComp'] = b_minNumAtomsForTargetComp
		MovePlans = selectCompsForBirthAtCurrentBatch(
			SS=SS, SSbatch=SSbatch,
			MovePlans=dict(),
			MoveRecordsByUID=MoveRecordsByUID,
			**BArgs)
		nChosen = len(MovePlans['b_targetUIDs'])
		nFailPerUID = list()
		for uid in SS.uids:
			bIDs = MoveRecordsByUID[uid]['b_batchIDsWhoseProposalFailed']
			if isinstance(bIDs, set):
				nFailPerUID.append(len(bIDs))
			else:
				nFailPerUID.append(0)
		nFailPerUID = np.asarray(nFailPerUID)
		nExpected = np.sum(np.logical_and(
			SS.N >= b_minNumAtomsForTargetComp,
			nFailPerUID < 1))
		assert nChosen == nExpected


def test_BPlanner_makePlanAtBatch_someDQForPrevFailuresWithOtherBatches(K=20):
	print('')
	SS = SuffStatBag(K=K)
	SS.setField('N',  np.arange(K), dims='K')
	SSbatch = SS.copy()

	# Select some subset of uids to be disqualified
	PRNG = np.random.RandomState(11)
	dqUIDs = PRNG.choice(K, size=3, replace=False)
	otherfailUIDs = PRNG.choice(K, size=3, replace=False)

	# Do the same test, while eliminating some uids
	MoveRecordsByUID = defaultdict(lambda: defaultdict(int))
	for uid in dqUIDs:
		MoveRecordsByUID[uid]['b_nFail'] = 1
		MoveRecordsByUID[uid]['b_nFailRecent'] = 1
		MoveRecordsByUID[uid]['b_batchIDsWhoseProposalFailed'] = set([0])
		print('PREV FAIL AT THIS BATCH: uid ', uid)
	for uid in otherfailUIDs:
		if uid in dqUIDs:
			continue
		MoveRecordsByUID[uid]['b_nFail'] = 1
		MoveRecordsByUID[uid]['b_nFailRecent'] = 1
		MoveRecordsByUID[uid]['b_batchIDsWhoseProposalFailed'] = set([1])
		print('PREV FAIL AT ANOTHER BATCH: uid ', uid)

	for b_minNumAtomsForTargetComp in [2, 5, 10, K]:
		BArgs['b_minNumAtomsForTargetComp'] = b_minNumAtomsForTargetComp
		MovePlans = selectCompsForBirthAtCurrentBatch(
			SS=SS, SSbatch=SSbatch,
			MovePlans=dict(),
			MoveRecordsByUID=MoveRecordsByUID,
			**BArgs)
		nChosen = len(MovePlans['b_targetUIDs'])
		nFailPerUID = list()
		for uid in SS.uids:
			bIDs = MoveRecordsByUID[uid]['b_batchIDsWhoseProposalFailed']
			if isinstance(bIDs, set) and 0 in bIDs:
					nFailPerUID.append(len(bIDs))
			else:
				nFailPerUID.append(0)
		nFailPerUID = np.asarray(nFailPerUID)
		nExpected = np.sum(np.logical_and(
			SS.N >= b_minNumAtomsForTargetComp,
			nFailPerUID < 1))
		assert nChosen == nExpected

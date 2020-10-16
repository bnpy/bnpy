'''
MOVBAlg.py

Implementation of Memoized Online VB (moVB) learn alg for bnpy models
'''
from collections import defaultdict
import numpy as np
import os
import logging

from MOVBAlg import MOVBAlg, makeDictOfAllWorkspaceVars
from bnpy.suffstats import SuffStatBag
from bnpy.util import isEvenlyDivisibleFloat
from bnpy.birthmove import TargetPlanner, TargetDataSampler, BirthMove
from bnpy.birthmove import BirthLogger
from bnpy.mergemove import MergeMove, MergePlanner, MergeLogger
from bnpy.birthmove.TargetDataSampler import hasValidKey
from bnpy.deletemove import DPlanner, DCollector, DEvaluator
from bnpy.deletemove import DeleteLogger


class MOVBBirthMergeAlg(MOVBAlg):

    def __init__(self, **kwargs):
        ''' Construct memoized algorithm instance that can do births/merges.
        '''
        MOVBAlg.__init__(self, **kwargs)

        if self.hasMove('merge') or self.hasMove('softmerge'):
            self.MergeLog = list()
            self.lapLastAcceptedMerge = self.algParams['startLap']

        if self.hasMove('birth'):
            # Track the number of laps since birth last attempted
            #  at each component, to encourage trying diversity
            self.LapsSinceLastBirth = defaultdict(int)
            self.BirthRecordsByComp = defaultdict(lambda: dict())

        if self.hasMove('delete'):
            self.DeleteRecordsByComp = defaultdict(lambda: dict())
            self.lapLastAcceptedDelete = self.algParams['startLap']

        if self.hasMove('seqcreate'):
            self.CreateRecords = defaultdict(lambda: dict())
        self.ELBOReady = True

    def fit(self, hmodel, DataIterator):
        ''' Run learning algorithm that fits parameters of hmodel to Data.

        Returns
        --------
        Info : dict of run information.

        Post Condition
        --------
        hmodel updated in place with improved global parameters.
        '''
        origmodel = hmodel
        self.ActiveIDVec = np.arange(hmodel.obsModel.K)
        self.maxUID = self.ActiveIDVec.max()
        self.DataIterator = DataIterator

        # Initialize progress tracking vars like nBatch, lapFrac, etc.
        iterid, lapFrac = self.initProgressTrackVars(DataIterator)

        # Keep list of params that should be retained across laps
        mkeys = hmodel.allocModel.get_keys_for_memoized_local_params()
        self.memoLPkeys = mkeys

        # Save initial state
        self.saveParams(lapFrac, hmodel)

        # Custom func hook
        self.eval_custom_func(
            isInitial=1, **makeDictOfAllWorkspaceVars(**vars()))

        # Prep for birth
        BirthPlans = list()
        BirthResults = list()
        prevBirthResults = list()

        # Prep for merge
        MergePlanInfo = dict()
        if self.hasMove('merge'):
            mergeStartLap = self.algParams['merge']['mergeStartLap']
        else:
            mergeStartLap = 0

        # Prep for delete
        DeletePlans = list()

        # Prep for shuffle
        order = None

        # Begin loop over batches of data...
        SS = None
        evBound = None
        isConverged = False
        self.set_start_time_now()
        while DataIterator.has_next_batch():

            # Grab new data
            Dchunk = DataIterator.get_next_batch()
            batchID = DataIterator.batchID
            Dchunk.batchID = batchID

            # Update progress-tracking variables
            iterid += 1
            lapFrac = (iterid + 1) * self.lapFracInc
            self.lapFrac = lapFrac
            nLapsCompleted = lapFrac - self.algParams['startLap']
            self.set_random_seed_at_lap(lapFrac)
            if self.doDebugVerbose():
                self.print_msg('========================== lap %.2f batch %d'
                               % (lapFrac, batchID))

            # Delete : Evaluate all planned proposals
            if self.isFirstBatch(lapFrac) and self.hasMove('delete'):
                self.DeleteAcceptRecord = dict()
                if self.doDebug() and len(DeletePlans) > 0:
                    DeletePlans[0]['WholeDataset'] = DataIterator.Data
                hmodel, SS = self.deleteAndUpdateMemory(hmodel, SS,
                                                        DeletePlans)
                DeletePlans = list()

            # Birth move : track birth info from previous lap
            if self.isFirstBatch(lapFrac):
                didBirth = self.do_birth_at_lap(lapFrac - 1.0)
                if self.hasMove('birth') and didBirth:
                    prevBirthResults = BirthResults
                else:
                    prevBirthResults = list()

            # Birth move : create new components
            if self.hasMove('birth') and self.do_birth_at_lap(lapFrac):
                if self.doBirthWithPlannedData(lapFrac):
                    hmodel, SS, BirthResults = self.birth_create_new_comps(
                        hmodel, SS, BirthPlans,
                        lapFrac=lapFrac)

                if self.doBirthWithDataFromCurrentBatch(lapFrac):
                    hmodel, SS, curResults = self.birth_create_new_comps(
                        hmodel, SS, Data=Dchunk,
                        lapFrac=lapFrac)
                    BirthResults.extend(curResults)
            else:
                BirthResults = list()

            # Prepare for merges
            if self.hasMove('merge') and self.doMergePrepAtLap(lapFrac):
                MergePrepInfo = self.preparePlansForMerge(
                    hmodel, SS, MergePrepInfo,
                    order=order,
                    BirthResults=BirthResults,
                    lapFrac=lapFrac)
            elif self.isFirstBatch(lapFrac):
                if self.doMergePrepAtLap(lapFrac + 1):
                    mpSelect = self.algParams['merge']['mergePairSelection']
                    MergePrepInfo = dict(mergePairSelection=mpSelect)
                else:
                    MergePrepInfo = dict()

            # Delete : Prepare plan for next lap.
            if self.hasMove('delete') and self.doDeleteAtLap(lapFrac + 1):
                if self.isFirstBatch(lapFrac) and SS is not None:
                    Plan = DPlanner.makePlanForEligibleComps(
                        SS,
                        lapFrac=self.lapFrac,
                        DRecordsByComp=self.DeleteRecordsByComp,
                        **self.algParams['delete'])
                    if 'candidateUIDs' in Plan:
                        DeletePlans = [Plan]
                    else:
                        DeletePlans = []

            # Reset selection terms to zero
            if self.isFirstBatch(lapFrac):
                if SS is not None and SS.hasSelectionTerms():
                    SS._SelectTerms.setAllFieldsToZero()

            # Local/E step
            self.algParamsLP['lapFrac'] = lapFrac
            self.algParamsLP['batchID'] = batchID

            if self.hasMove('seqcreate'):
                LPchunk = self.localStepWithBirthAtEachSeq(
                    hmodel, SS, Dchunk, batchID,
                    lapFrac=lapFrac,
                    evBound=evBound,
                    **MergePrepInfo)
            else:
                LPchunk = self.memoizedLocalStep(hmodel, Dchunk, batchID,
                                                 **MergePrepInfo)

            # Summary step
            SS, SSchunk = self.memoizedSummaryStep(hmodel, SS,
                                                   Dchunk, LPchunk, batchID,
                                                   MergePrepInfo=MergePrepInfo,
                                                   )

            # Delete : Collect target dataset
            if len(DeletePlans) > 0:
                DeletePlans = self.deleteCollectTarget(Dchunk, hmodel, LPchunk,
                                                       batchID,
                                                       DeletePlans)

            # Birth move : collect target data
            if self.hasMove('birth') and self.do_birth_at_lap(lapFrac + 1.0):
                if self.isFirstBatch(lapFrac):
                    BirthPlans = self.birth_plan_targets_for_next_lap(
                        Dchunk, hmodel, SS, LPchunk, BirthResults)
                BirthPlans = self.birth_collect_target_subsample(
                    Dchunk, hmodel, LPchunk, BirthPlans, lapFrac)
            else:
                BirthPlans = list()

            # Birth : Handle removing "extra mass" of fresh components
            if self.hasMove('birth') and self.isLastBatch(lapFrac):
                hmodel, SS = self.birth_remove_extra_mass(
                    hmodel, SS, BirthResults)
                # SS now has size exactly consistent with entire dataset

            # Global/M step
            self.GlobalStep(hmodel, SS, lapFrac)

            # ELBO calculation
            if self.isLastBatch(lapFrac):
                # after seeing all data, ELBO will be ready
                self.ELBOReady = True
            if self.ELBOReady:
                evBound = hmodel.calc_evidence(SS=SS)

            # Merge move!
            if self.hasMove('merge') and self.isLastBatch(lapFrac) \
                    and lapFrac > mergeStartLap:
                hmodel, SS, evBound = self.run_many_merge_moves(
                    hmodel, SS, evBound, lapFrac, MergePrepInfo)
                # Cancel all planned deletes if merges were accepted.
                if hasattr(self, 'MergeLog') and len(self.MergeLog) > 0:
                    DeletePlans = []
                    # Update memoized stats for each batch
                    self.fastForwardMemory(Kfinal=SS.K)
                    if hasattr(SS, 'mPairIDs'):
                        del SS.mPairIDs

            # Shuffle : Rearrange topic order (big to small)
            didFastFwd = 0
            if self.hasMove('shuffle') and self.isLastBatch(lapFrac):
                order = np.argsort(-1 * SS.getCountVec())
                sortedalready = np.arange(SS.K)
                if np.allclose(order, sortedalready):
                    order = None  # Already sorted, do nothing!
                else:
                    self.ActiveIDVec = self.ActiveIDVec[order]
                    SS.reorderComps(order)
                    assert np.allclose(SS.uIDs, self.ActiveIDVec)
                    hmodel.update_global_params(SS)
                    evBound = hmodel.calc_evidence(SS=SS)
                    # Update tracked target stats for any upcoming deletes
                    for DPlan in DeletePlans:
                        if self.hasMove('merge'):
                            assert len(self.MergeLog) == 0
                        Kextra = SS.K - DPlan['targetSS'].K
                        if Kextra > 0:
                            delattr(DPlan['targetSS'], 'uIDs')
                            DPlan['targetSS'].insertEmptyComps(Kextra)
                        DPlan['targetSS'].reorderComps(order)
                        DPlan['targetSS'].uIDs = self.ActiveIDVec.copy()
                        targetSSbyBatch = DPlan['targetSSByBatch']
                        for batchID in targetSSbyBatch:
                            batchSS = targetSSbyBatch[batchID]
                            Kextra = SS.K - batchSS.K
                            if Kextra > 0:
                                delattr(batchSS, 'uIDs')
                                batchSS.insertEmptyComps(Kextra)
                            batchSS.reorderComps(order)
                            batchSS.uIDs = self.ActiveIDVec.copy()
                    # Update memoized stats for each batch
                    self.fastForwardMemory(Kfinal=SS.K, order=order)
                    didFastFwd = 1

            if self.hasMove('seqcreate') and self.isLastBatch(lapFrac):
                if not didFastFwd:
                    Kr, Kextra = self.CreateRecords[np.ceil(lapFrac)]
                    if Kextra > 0:
                        self.fastForwardMemory(Kfinal=SS.K)
                        didFastFwd = 1

            if nLapsCompleted > 1.0 and len(BirthResults) == 0:
                # evBound increases monotonically AFTER first lap
                # verify_evidence warns if this isn't happening
                self.verify_evidence(evBound, prevBound, lapFrac)

            if self.doDebug() and lapFrac >= 1.0:
                self.verifyELBOTracking(hmodel, SS, evBound,
                                        BirthResults=BirthResults)

            # Assess convergence
            countVec = SS.getCountVec()
            if lapFrac > 1.0 and self.isLastBatch(lapFrac):
                isConverged = self.isCountVecConverged(countVec, prevCountVec)
                hasMoreMoves = self.hasMoreReasonableMoves(lapFrac, SS)
                doneRequiredLaps = nLapsCompleted >= self.algParams['minLaps']
                isConverged = isConverged and not hasMoreMoves \
                    and doneRequiredLaps
                self.setStatus(lapFrac, isConverged)

            # Display progress
            self.updateNumDataProcessed(Dchunk.get_size())
            if self.isLogCheckpoint(lapFrac, iterid):
                self.printStateToLog(hmodel, evBound, lapFrac, iterid)

            # Save diagnostics and params
            if self.isSaveDiagnosticsCheckpoint(lapFrac, iterid):
                self.saveDiagnostics(lapFrac, SS, evBound, self.ActiveIDVec)
            if self.isSaveParamsCheckpoint(lapFrac, iterid):
                self.saveParams(lapFrac, hmodel, SS)

            # Custom func hook
            self.eval_custom_func(**makeDictOfAllWorkspaceVars(**vars()))

            if isConverged and self.isLastBatch(lapFrac):
                break
            prevCountVec = countVec.copy()
            prevBound = evBound
            # .... end loop over data

        # Finished! Save, print and exit
        self.printStateToLog(hmodel, evBound, lapFrac, iterid, isFinal=1)
        self.saveParams(lapFrac, hmodel, SS)
        self.eval_custom_func(
            isFinal=1, **makeDictOfAllWorkspaceVars(**vars()))

        # Births and merges require copies of original model object
        #  we need to make sure original reference has updated parameters, etc.
        if id(origmodel) != id(hmodel):
            origmodel.allocModel = hmodel.allocModel
            origmodel.obsModel = hmodel.obsModel
        return self.buildRunInfo(evBound=evBound, SS=SS,
                                 LPmemory=self.LPmemory,
                                 SSmemory=self.SSmemory)

    def hasMoreReasonableMoves(self, lapFrac, SS):
        ''' Decide if more moves will feasibly change current configuration.
        '''
        if lapFrac - self.algParams['startLap'] >= self.algParams['nLap']:
            # Time's up, so doesn't matter what other moves are possible.
            return False

        if self.hasMove('delete'):
            # If any eligible comps exist, we have more moves possible
            # so return True
            deleteStartLap = self.algParams['delete']['deleteStartLap']
            nBeforeQuit = self.algParams['delete']['deleteNumStuckBeforeQuit']
            waitedLongEnough = (
                lapFrac - self.lapLastAcceptedDelete) > nBeforeQuit

            # If we haven't tried deletes for sufficent num laps, keep going
            if lapFrac <= deleteStartLap + nBeforeQuit:
                return True

            if isEvenlyDivisibleFloat(lapFrac, 1.0):
                nEligible = DPlanner.getEligibleCount(
                    SS,
                    DRecordsByComp=self.DeleteRecordsByComp,
                    **self.algParams['delete'])
            else:
                nEligible = 1

            if nEligible > 0 or not waitedLongEnough:
                return True

        if self.hasMove('birth') and self.do_birth_at_lap(lapFrac):
            # If any eligible comps exist, we have more moves possible
            # so return True
            if not hasattr(self, 'BirthEligibleHist'):
                return True
            if self.BirthEligibleHist['Nable'] > 0:
                return True

        if self.hasMove('merge'):
            nStuckBeforeQuit = self.algParams[
                'merge']['mergeNumStuckBeforeQuit']
            mergeStartLap = self.algParams['merge']['mergeStartLap']

            # If we haven't tried merges for sufficent num laps, keep going
            if lapFrac <= mergeStartLap + nStuckBeforeQuit:
                return True

            # If we've tried many merges without success, exit early
            if (lapFrac - self.lapLastAcceptedMerge) > nStuckBeforeQuit:
                return False
            return True

        return False
        # ... end function hasMoreReasonableMoves

    def localStepWithBirthAtEachSeq(self, hmodel, SS, Dchunk, batchID,
                                    lapFrac=0,
                                    evBound=None,
                                    **LPandMergeKwargs):
        ''' Do local step on provided data chunk, possibly making new states.

        Returns
        -------
        LPchunk : dict of local params for each seq in current batch
            Number of states will be greater or equal to hmodel.obsModel.K
        '''
        from bnpy.init.SingleSeqStateCreator import \
            createSingleSeqLPWithNewStates_ManyProposals
        from bnpy.init.SeqCreateRefinery import \
            deleteEmptyCompsAndKeepConsistentWithWholeDataset
        if lapFrac <= self.algParams['seqcreate']['creationLapDelim_early']:
            Kfresh = self.algParams['seqcreate']['creationKfresh_early']
            numProp = self.algParams['seqcreate']['creationNumProposal_early']

        elif lapFrac > self.algParams['seqcreate']['creationLapDelim_late']:
            Kfresh = 0
            numProp = 0
        else:
            Kfresh = self.algParams['seqcreate']['creationKfresh_late']
            numProp = self.algParams['seqcreate']['creationNumProposal_late']

        seqcreateParams = dict(**self.algParams['seqcreate'])
        seqcreateParams['creationNumProposal'] = numProp
        seqcreateParams['Kfresh'] = Kfresh
        seqcreateParams['PRNG'] = self.PRNG

        if lapFrac <= 1.0 and self.algParams['doFullPassBeforeMstep']:
            Kfresh = 0

        Korig = hmodel.obsModel.K
        tempModel = hmodel.copy()
        if SS is None:
            tempSS = None
        else:
            tempSS = SS.copy(includeELBOTerms=1, includeMergeTerms=0)

        if lapFrac > 1:
            assert np.allclose(SS.nDoc, Dchunk.nDocTotal)

        didAnyProposals = False
        if Kfresh > 0 and numProp > 0:
            # NOTE: Here, make sure SS retains all information from
            # *ALL* previously seen batches, including the current one.
            # Otherwise, could wipe out special states and lose guarantees.

            if self.isFirstBatch(lapFrac):
                if not hasattr(self, 'SeqCreatePastAttemptLog'):
                    self.SeqCreatePastAttemptLog = dict()

                elif 'nTryByStateUID' in self.SeqCreatePastAttemptLog:
                    for uID in self.SeqCreatePastAttemptLog['nTryByStateUID']:
                        if uID not in self.ActiveIDVec:
                            continue
                        print(' uid %d: nTry %d' % (
                            uID,
                            self.SeqCreatePastAttemptLog['nTryByStateUID'][uID]
                            ))
                    print('^^^^^^^^^^^^^^^^^^^^^^^^^^')
                self.SeqCreatePastAttemptLog['maxUID'] = 1 * self.maxUID
                self.SeqCreatePastAttemptLog['uIDs'] = self.ActiveIDVec.copy()

            if SS is None or self.isFirstBatch(lapFrac):
                self.nDocSeenInCurLap = 0

            randOrder = self.PRNG.permutation(np.arange(Dchunk.nDoc))
            for orderID, n in enumerate(randOrder):
                # Track num docs we've done a proposal before current n.
                nDocSeenForProposal = orderID

                seqName = "seqUID %d | %d/%d in batch %d" % (
                    n, orderID + 1, Dchunk.nDoc, batchID)
                if seqcreateParams['doVizSeqCreate']:
                    doTrackTruth = 1
                else:
                    doTrackTruth = 0
                Data_n = Dchunk.select_subset_by_mask(
                    [n], doTrackTruth=doTrackTruth)
                LP_n = tempModel.calc_local_params(Data_n, **self.algParamsLP)
                LP_n, tempModel, tempSS, Info = \
                    createSingleSeqLPWithNewStates_ManyProposals(
                        Data_n, LP_n, tempModel, SS=tempSS,
                        lapFrac=lapFrac,
                        nDocSeenForProposal=orderID,
                        nDocSeenInCurLap=self.nDocSeenInCurLap,
                        batchID=batchID,
                        seqName=seqName,
                        PastAttemptLog=self.SeqCreatePastAttemptLog,
                        n=n,
                        **seqcreateParams)
                SS_n = tempModel.get_global_suff_stats(Data_n, LP_n)
                if orderID == 0:
                    SSchunk = SS_n
                else:
                    Kextra = SS_n.K - SSchunk.K
                    if Kextra > 0:
                        SSchunk.insertEmptyComps(Kextra)
                    SSchunk += SS_n

                if Info['didAnyProposals']:
                    if lapFrac > 1:
                        nDocTotal = Data_n.nDocTotal
                        assert tempSS.nDoc == nDocTotal + \
                            nDocSeenForProposal + 1
                    else:
                        assert tempSS.nDoc == nDocSeenForProposal + 1 + \
                            self.nDocSeenInCurLap
                didAnyProposals = didAnyProposals or Info['didAnyProposals']

            assert SSchunk.nDoc == Dchunk.nDoc
            assert SSchunk.K == tempSS.K
            assert tempSS.nDoc >= SSchunk.nDoc

            # Remove any states that are empty and unique to this batch
            # since this is better than letting them stick around until later
            LPchunk, SSchunk, tempModel, tempSS, Info = \
                deleteEmptyCompsAndKeepConsistentWithWholeDataset(
                    Dchunk, SSchunk, tempModel, tempSS,
                    origK=Korig)

        # Track total docs seen thus far,
        # to be sure SS bookkeeping is done correctly
        if didAnyProposals:
            self.nDocSeenInCurLap += Dchunk.nDoc

        # Do final analysis of all sequences in this chunk
        # so that every sequence can use every newfound state
        LPandMergeKwargs.update(self.algParamsLP)
        LPchunk = tempModel.calc_local_params(Dchunk, **LPandMergeKwargs)
        SSchunk = tempModel.get_global_suff_stats(Dchunk, LPchunk,
                                                  doPrecompEntropy=1)

        if didAnyProposals and lapFrac > 1:
            assert SS != tempSS
            assert SS.nDoc == Dchunk.nDocTotal
            assert tempModel != hmodel

            # Use most recent estimate of whole-dataset ELBO on current model
            curELBO = evBound * 1.0
            assert np.isfinite(curELBO)

            print('<<<<<<<<<<<<<<<<<<<<< lap %.2f batch %d' % (
                lapFrac, batchID))
            print('------- BEFORE K=%d' % (SS.K))
            print('Whole N ', ' '.join(
                ['%5.1f' % (x) for x in SS.N[:20]]))
            print('Batch N ', ' '.join(
                ['%5.1f' % (x) for x in self.SSmemory[batchID].N[:20]]))

            # Compute whole-dataset ELBO on proposed model
            # Adjusting suff stats to exactly represent whole dataset
            # including new assignments of current batch
            delattr(tempModel.allocModel, 'rho')
            tempSS = SS.copy(includeELBOTerms=1, includeMergeTerms=0)
            prevSSchunk = self.SSmemory[batchID].copy(includeELBOTerms=1,
                                                      includeMergeTerms=0)
            Kextra = tempSS.K - prevSSchunk.K
            if Kextra > 0:
                prevSSchunk.insertEmptyComps(Kextra)
            tempSS -= prevSSchunk
            Kextra = SSchunk.K - tempSS.K
            if Kextra > 0:
                tempSS.insertEmptyComps(Kextra)
            # Add in newest stats for this batch, which might have new states
            assert SSchunk.hasELBOTerms()
            assert tempSS.hasELBOTerms()
            tempSS += SSchunk
            tempModel.update_global_params(tempSS)
            # Now, compute the whole-dataset objective
            assert tempSS.nDoc == Dchunk.nDocTotal
            propELBO = tempModel.calc_evidence(SS=tempSS)
            assert np.isfinite(propELBO)

            print('------- AFTER K=%d' % (tempSS.K))
            print('Whole N ', ' '.join(
                ['%5.1f' % (x) for x in tempSS.N[:20]]))
            print('Batch N ', ' '.join(
                ['%5.1f' % (x) for x in SSchunk.N[:20]]))

            print('curELBO  % .7f' % (curELBO))
            print('propELBO % .7f' % (propELBO))
            if propELBO > curELBO:
                print('ACCEPTED!')
            else:
                print('rejected')
                LPchunk = hmodel.calc_local_params(Dchunk, **LPandMergeKwargs)

        Kresult = LPchunk['resp'].shape[1]
        Kextra = Kresult - Korig
        if not self.isFirstBatch(lapFrac):
            Kr, Kx_prev = self.CreateRecords[np.ceil(lapFrac)]
            Kextra += Kx_prev
        self.CreateRecords[np.ceil(lapFrac)] = (Kresult, Kextra)
        return LPchunk

        ''' EDIT this chunk is superceded by call to deleteAndKeepConsistent
        emptyIDs = Korig + np.flatnonzero(SSchunk.N[Korig:] <= 1)
        if len(emptyIDs) > 0:
            print 'REALLY SHOULD DELETE EMPTIES HERE'
        Kresult = LPchunk['resp'].shape[1]
        Kextra = Kresult - Korig
        while len(emptyIDs) > 0:
            for kempty in reversed(emptyIDs):
                SSchunk.removeComp(kempty)
                Kextra -= 1
            # tempSS needs to be a consistent set of stats
            # that just might double count current chunk after the first lap
            if SS is None:
                tempSS = SSchunk
            else:
                tempSS = SS.copy()
                tempSS.insertEmptyComps(Kextra)
                tempSS += SSchunk

            tempModel.update_global_params(tempSS)
            LPchunk = tempModel.calc_local_params(Dchunk, **LPandMergeKwargs)
            SSchunk = tempModel.get_global_suff_stats(Dchunk, LPchunk)
            emptyIDs = Korig + np.flatnonzero(SSchunk.N[Korig:] <= 1)
        '''

    def memoizedLocalStep(self, hmodel, Dchunk, batchID, **LPandMergeKwargs):
        ''' Execute local step on data chunk.

        Returns
        --------
        LPchunk : dict of local params for current batch
        '''
        if batchID in self.LPmemory:
            oldLPchunk = self.load_batch_local_params_from_memory(batchID)
        else:
            oldLPchunk = None

        LPandMergeKwargs.update(self.algParamsLP)
        LPchunk = hmodel.calc_local_params(Dchunk, oldLPchunk,
                                           **LPandMergeKwargs)
        if self.algParams['doMemoizeLocalParams']:
            self.save_batch_local_params_to_memory(batchID, LPchunk)
        return LPchunk

    def load_batch_local_params_from_memory(self, batchID):
        ''' Load local parameter dict stored in memory for provided batchID

        Ensures "fast-forward" so that all recent merges/births
        are accounted for in the returned LP

        Returns
        -------
        LPchunk : bnpy local parameters dictionary for batchID
        '''
        LPchunk = self.LPmemory[batchID]
        if self.hasMove('merge') and LPchunk is not None:
            K = LPchunk[self.memoLPkeys[0]].shape[1]
            for MInfo in self.MergeLog:
                kA = MInfo['kA']
                kB = MInfo['kB']
                for key in self.memoLPkeys:
                    if kA >= K or kB >= K:
                        # Stored LPchunk is outdated... forget it.
                        return None
                    kB_column = LPchunk[key][:, kB]
                    LPchunk[key] = np.delete(LPchunk[key], kB, axis=1)
                    LPchunk[key][:, kA] = LPchunk[key][:, kA] + kB_column

        return LPchunk

    def save_batch_local_params_to_memory(self, batchID, LPchunk, doCopy=0):
        ''' Store certain local params into internal LPmemory cache.

        Fields to save determined by the memoLPkeys attribute of this alg.

        Returns
        ---------
        None. self.LPmemory updated in-place.
        '''
        keepLPchunk = dict()
        for key in list(LPchunk.keys()):
            if key in self.memoLPkeys:
                if doCopy:
                    keepLPchunk[key] = copy.deepcopy(LPchunk[key])
                else:
                    keepLPchunk[key] = LPchunk[key]

        if len(list(keepLPchunk.keys())) > 0:
            self.LPmemory[batchID] = keepLPchunk
        else:
            self.LPmemory[batchID] = None

    def memoizedSummaryStep(self, hmodel, SS, Dchunk, LPchunk, batchID,
                            MergePrepInfo=None):
        ''' Execute summary step on current batch and update aggregated SS

        Returns
        --------
        SS : updated aggregate suff stats
        SSchunk : updated current-batch suff stats
        '''
        if MergePrepInfo is None:
            MergePrepInfo = dict()
        if batchID in self.SSmemory:
            # Decrement old value of SSchunk from aggregated SS
            # oldSSchunk will have usual Fields and ELBOTerms,
            # but all MergeTerms and SelectionTerms should be removed.
            oldSSchunk = self.load_batch_suff_stat_from_memory(
                batchID, doCopy=0, Kfinal=SS.K)
            assert not oldSSchunk.hasMergeTerms()
            assert oldSSchunk.K == SS.K
            assert np.allclose(SS.uIDs, oldSSchunk.uIDs)
            SS -= oldSSchunk

        # Calculate fresh suff stats for current batch
        if self.hasMove('delete'):
            trackDocUsage = 1
        else:
            trackDocUsage = 0
        SSchunk = hmodel.get_global_suff_stats(Dchunk, LPchunk,
                                               doPrecompEntropy=1,
                                               trackDocUsage=trackDocUsage,
                                               **MergePrepInfo)
        if SSchunk.K > self.ActiveIDVec.size:
            for newCompID in np.arange(self.ActiveIDVec.size, SSchunk.K):
                self.maxUID += 1
                self.ActiveIDVec = np.hstack([
                    self.ActiveIDVec, self.maxUID])
        SSchunk.setUIDs(self.ActiveIDVec.copy())
        assert np.allclose(SSchunk.uIDs, self.ActiveIDVec)

        # Increment aggregated SS by adding in SSchunk
        if SS is None:
            SS = SSchunk.copy()
        else:
            if SS.K < SSchunk.K:
                # Catch up by added empty comps
                SS.insertEmptyComps(SSchunk.K - SS.K)
                SS.setUIDs(self.ActiveIDVec.copy())

            assert SSchunk.K == SS.K
            assert np.allclose(SS.uIDs, self.ActiveIDVec)
            SS += SSchunk
            if not SS.hasSelectionTerms() and SSchunk.hasSelectionTerms():
                SS._SelectTerms = SSchunk._SelectTerms

        self.save_batch_suff_stat_to_memory(batchID, SSchunk)

        # Force aggregated suff stats to obey required constraints.
        # This avoids numerical issues caused by incremental updates
        if hasattr(hmodel.allocModel, 'forceSSInBounds'):
            hmodel.allocModel.forceSSInBounds(SS)
        if hasattr(hmodel.obsModel, 'forceSSInBounds'):
            hmodel.obsModel.forceSSInBounds(SS)
        return SS, SSchunk

    def load_batch_suff_stat_from_memory(self, batchID, doCopy=0,
                                         Kfinal=0, order=None):
        ''' Load (fast-forwarded) suff stats from previous visit to batchID.

        Any merges, shuffles, or births which happened since last visit
        are automatically applied.

        Returns
        -------
        SSchunk : bnpy SuffStatDict object for batchID

        Post Condition
        --------------
        SSchunk has same ActiveUIDs as self.
        SSchunk has no merge terms at all.
        '''
        SSchunk = self.SSmemory[batchID]
        if doCopy:
            # Duplicating to avoid changing the raw data stored in SSmemory
            #  this is done usually when debugging.
            SSchunk = SSchunk.copy()

        # Check to see if we've fast-forwarded this chunk already
        # If so, we return as-is
        if SSchunk.K == self.ActiveIDVec.size:
            if np.allclose(SSchunk.uIDs, self.ActiveIDVec):
                if SSchunk.hasMergeTerms():
                    SSchunk.removeMergeTerms()
                if hasattr(SSchunk, 'mPairIDs'):
                    del SSchunk.mPairIDs
                return SSchunk

        # Fast-forward accepted softmerges from end of previous lap
        if self.hasMove('softmerge'):
            for MInfo in self.MergeLog:
                SSchunk.multiMergeComps(MInfo['kdel'], MInfo['alph'])

        # Fast-forward accepted merges from end of previous lap
        if self.hasMove('merge') and SSchunk.hasMergeTerms():
            for MInfo in self.MergeLog:
                kA = MInfo['kA']
                kB = MInfo['kB']
                if kA < SSchunk.K and kB < SSchunk.K:
                    SSchunk.mergeComps(kA, kB)
        if SSchunk.hasMergeTerms():
            SSchunk.removeMergeTerms()
        if hasattr(SSchunk, 'mPairIDs'):
            del SSchunk.mPairIDs

        # Fast-forward births from this lap
        if self.hasMove('seqcreate') and Kfinal > 0 and SSchunk.K < Kfinal:
            Kextra = Kfinal - SSchunk.K
            curUIDs = SSchunk.uIDs
            if Kextra > 0:
                SSchunk.insertEmptyComps(Kextra)
                newUIDs = np.arange(self.maxUID - Kextra + 1, self.maxUID + 1)
                SSchunk.setUIDs(np.hstack([curUIDs, newUIDs]))
            assert SSchunk.K == Kfinal

        # Fast-forward any shuffling/reordering that happened
        if self.hasMove('shuffle') and order is not None:
            if len(order) == SSchunk.K:
                SSchunk.reorderComps(order)
            else:
                msg = 'Order has wrong size.'
                msg += '\n size order  : %d' % len(order)
                msg += '\n size SSchunk: %d' % SSchunk.K
                raise ValueError(msg)

        isGoodSize = SSchunk.uIDs.size == SSchunk.K
        if not isGoodSize:
            if self.algParams['debug'] == 'interactive':
                from IPython import embed
                embed()
        assert isGoodSize

        isGood = np.allclose(SSchunk.uIDs, self.ActiveIDVec[:SSchunk.K])
        if not isGood:
            if self.algParams['debug'] == 'interactive':
                from IPython import embed
                embed()
        assert isGood

        # Fast-forward births from this lap
        if self.hasMove('birth') and Kfinal > 0 and SSchunk.K < Kfinal:
            Kextra = Kfinal - SSchunk.K
            if Kextra > 0:
                SSchunk.insertEmptyComps(Kextra)
            assert SSchunk.K == Kfinal
            SSchunk.setUIDs(self.ActiveIDVec.copy())

        assert np.allclose(SSchunk.uIDs, self.ActiveIDVec)
        return SSchunk

    def save_batch_suff_stat_to_memory(self, batchID, SSchunk):
        ''' Store the provided suff stats into the "memory" for later retrieval
        '''
        if SSchunk.hasSelectionTerms():
            del SSchunk._SelectTerms
        self.SSmemory[batchID] = SSchunk

    def fastForwardMemory(self, Kfinal=0, order=None):
        ''' Update *every* batch in memory to be current
        '''
        for batchID in self.SSmemory:
            self.load_batch_suff_stat_from_memory(
                batchID, Kfinal=Kfinal, order=order)

    def doBirthWithPlannedData(self, lapFrac):
        return self.isFirstBatch(lapFrac)

    def doBirthWithDataFromCurrentBatch(self, lapFrac):
        if self.isLastBatch(lapFrac):
            return False
        rem = lapFrac - np.floor(lapFrac)
        isWithinFrac = rem <= self.algParams['birth']['birthBatchFrac'] + 1e-6
        isWithinLimit = lapFrac <= self.algParams[
            'birth']['birthBatchLapLimit']
        return isWithinFrac and isWithinLimit

    def birth_create_new_comps(self, hmodel, SS, BirthPlans=list(), Data=None,
                               lapFrac=0):
        ''' Create new components

            Returns
            -------
            hmodel : bnpy HModel, either existing model or one with more comps
            SS : bnpy SuffStatBag, either existing SS or one with more comps
            BirthResults : list of dicts, one entry per birth move
        '''
        kwargs = dict(**self.algParams['birth'])
        kwargs.update(**self.algParamsLP)

        if 'birthRetainExtraMass' not in kwargs:
            kwargs['birthRetainExtraMass'] = 1

        if Data is not None:
            targetData, targetInfo = TargetDataSampler.sample_target_data(
                Data, model=hmodel, LP=None,
                randstate=self.PRNG,
                **kwargs)
            Plan = dict(Data=targetData, ktarget=-1, targetWordIDs=[-1])
            BirthPlans = [Plan]
            kwargs['birthRetainExtraMass'] = 0

        nMoves = len(BirthPlans)
        BirthResults = list()

        def isInPlan(Plan, key):
            return key in Plan and Plan[key] is not None

        for moveID, Plan in enumerate(BirthPlans):
            # Unpack data for current move
            ktarget = Plan['ktarget']
            targetData = Plan['Data']
            targetSize = TargetDataSampler.getSize(targetData)
            # Remember, targetData may be None

            if isInPlan(Plan, 'targetWordIDs'):
                isBad = len(Plan['targetWordIDs']) == 0
            elif isInPlan(Plan, 'targetWordFreq'):
                isBad = False
            else:
                isBad = ktarget is None

            BirthLogger.logStartMove(lapFrac, moveID + 1, len(BirthPlans))
            if isBad or targetData is None:
                msg = Plan['msg']
                BirthLogger.log(msg, 'moreinfo')
                BirthLogger.log('SKIPPED. TargetData bad.', 'moreinfo')
            elif targetSize < kwargs['Kfresh']:
                msg = "SKIPPED. Target data too small. Size %d, expected >= %d"
                msg = msg % (targetSize, kwargs['Kfresh'])
                BirthLogger.log(msg, 'moreinfo')
            else:
                newmodel, newSS, MoveInfo = BirthMove.run_birth_move(
                    hmodel, SS, targetData,
                    randstate=self.PRNG,
                    Plan=Plan,
                    **kwargs)
                hmodel = newmodel
                SS = newSS

                if MoveInfo['didAddNew']:
                    BirthResults.append(MoveInfo)
                    for kk in MoveInfo['birthCompIDs']:
                        self.LapsSinceLastBirth[kk] = -1

                        self.maxUID += 1
                        self.ActiveIDVec = np.append(
                            self.ActiveIDVec, self.maxUID)
                    SS.setUIDs(self.ActiveIDVec.copy())

                # Update BirthRecords to track comps that fail at births
                targetUID = Plan['targetUID']
                if MoveInfo['didAddNew']:
                    # Remove from records if successful... this comp will
                    # change a lot
                    if targetUID in self.BirthRecordsByComp:
                        del self.BirthRecordsByComp[targetUID]
                else:
                    if 'nFail' not in self.BirthRecordsByComp[targetUID]:
                        self.BirthRecordsByComp[targetUID]['nFail'] = 1
                    else:
                        self.BirthRecordsByComp[targetUID]['nFail'] += 1
                    self.BirthRecordsByComp[targetUID]['count'] = Plan['count']

        return hmodel, SS, BirthResults

    def birth_remove_extra_mass(self, hmodel, SS, BirthResults):
        ''' Adjust model and suff stats to remove extra mass from birth.

        After this call, SS should have scale exactly consistent with
        the entire dataset (all B batches).

        Returns
        -------
        hmodel : bnpy HModel
        SS : bnpy SuffStatBag
        '''
        didChangeSS = False
        for MoveInfo in BirthResults:
            if MoveInfo['didAddNew'] and 'extraSS' in MoveInfo:
                extraSS = MoveInfo['extraSS']
                compIDs = MoveInfo['modifiedCompIDs']
                assert extraSS.K == len(compIDs)
                SS.subtractSpecificComps(extraSS, compIDs)
                didChangeSS = True
                MoveInfo['extraSSDone'] = 1
        if didChangeSS:
            hmodel.update_global_params(SS)
        return hmodel, SS

    def birth_plan_targets_for_next_lap(self, Data, hmodel, SS, LP,
                                        BirthResults):
        ''' Create plans for next lap's birth moves

        Returns
        -------
        BirthPlans : list of dicts,
                     each entry represents the plan for one future birth move
        '''
        assert SS is not None
        assert hmodel.allocModel.K == SS.K
        K = hmodel.allocModel.K
        nBirths = self.algParams['birth']['birthPerLap']

        if self.algParams['birth']['targetSelectName'] == 'smart':
            if self.lapFrac < 1:
                ampF = Data.get_total_size() / float(Data.get_size())
            else:
                ampF = 1.0
            ampF = np.maximum(ampF, 1.0)
            Plans = TargetPlanner.makePlans_TargetCompsSmart(
                SS,
                self.BirthRecordsByComp,
                self.lapFrac,
                ampF=ampF,
                **self.algParams['birth'])
            Hist, CStatus, msg = self.birth_makeEligibilityHist(SS)
            self.BirthEligibleHist = Hist
            BirthLogger.logStartPrep(self.lapFrac + 1)
            BirthLogger.log(msg, 'moreinfo')

            SaveVars = dict()
            SaveVars['lapFrac'] = self.lapFrac
            SaveVars['msg'] = msg
            SaveVars['BirthEligibleHist'] = self.BirthEligibleHist

            savedict = dict()
            for compID in SS.uIDs:
                if compID in self.BirthRecordsByComp:
                    savedict[compID] = self.BirthRecordsByComp[compID]
            SaveVars['BirthRecordsByComp'] = savedict
            SaveVars['CompStatus'] = CStatus
            # # Uncomment in future when birth logging is more of a priority
            # import joblib
            # if self.savedir is not None:
            #    dumpfile = os.path.join(self.savedir, 'birth-plans.dump')
            #    joblib.dump(SaveVars, dumpfile)
            return Plans

        # Update counter for duration since last targeted-birth for each comp
        for kk in range(K):
            self.LapsSinceLastBirth[kk] += 1
        # Ignore components that have just been added to the model.
        excludeList = self.birth_get_all_new_comps(BirthResults)

        # For each birth move, create a "plan"
        BirthPlans = list()
        for posID in range(nBirths):
            try:
                ktarget, ps = TargetPlanner.select_target_comp(
                    K, SS=SS, Data=Data, model=hmodel,
                    randstate=self.PRNG,
                    excludeList=excludeList,
                    return_ps=1,
                    lapsSinceLastBirth=self.LapsSinceLastBirth,
                    **self.algParams['birth'])
                targetUID = self.ActiveIDVec[ktarget]

                self.LapsSinceLastBirth[ktarget] = 0
                excludeList.append(ktarget)
                Plan = dict(ktarget=ktarget,
                            targetUID=targetUID,
                            Data=None,
                            targetWordIDs=None,
                            targetWordFreq=None)
            except BirthMove.BirthProposalError as e:
                # Happens when no component is eligible for selection (all
                # excluded)
                Plan = dict(ktarget=None,
                            Data=None,
                            targetWordIDs=None,
                            targetWordFreq=None,
                            msg=str(e),
                            )
            BirthPlans.append(Plan)

        return BirthPlans

    def birth_collect_target_subsample(self, Dchunk, model, LPchunk,
                                       BirthPlans, lapFrac):
        ''' Collect subsample of the data in Dchunk, and add that subsample
              to overall targeted subsample stored in input list BirthPlans
            This overall sample is aggregated across many batches of data.
            Data from Dchunk is only collected if more data is needed.

            Returns
            -------
            BirthPlans : list of planned births for the next lap,
                          updated to include data from Dchunk if needed
        '''
        for Plan in BirthPlans:
            # Skip this move if component selection failed
            if Plan['ktarget'] is None and Plan['targetWordIDs'] is None \
                    and Plan['targetWordFreq'] is None:
                continue

            birthParams = dict(**self.algParams['birth'])

            # Skip collection if have enough data already
            if Plan['Data'] is not None:
                targetSize = TargetDataSampler.getSize(Plan['Data'])
                if targetSize >= birthParams['targetMaxSize']:
                    continue
                birthParams['targetMaxSize'] -= targetSize
                # TODO: worry about targetMaxSize when we always keep topK
                # datapoints

            # Sample data from current batch, if more is needed
            targetData, targetInfo = TargetDataSampler.sample_target_data(
                Dchunk, model=model, LP=LPchunk,
                targetCompID=Plan['ktarget'],
                targetWordIDs=Plan['targetWordIDs'],
                targetWordFreq=Plan['targetWordFreq'],
                randstate=self.PRNG,
                return_Info=True,
                **birthParams)

            # Update Data for current entry in self.targetDataList
            if targetData is None:
                if Plan['Data'] is None:
                    cmsg = "TargetData: No samples for target comp found."
                    Plan['msg'] = cmsg
            else:
                if Plan['Data'] is None:
                    Plan['Data'] = targetData
                    Plan['Info'] = targetInfo
                else:
                    Plan['Data'].add_data(targetData)
                    if 'dist' in Plan['Info']:
                        Plan['Info']['dist'] = np.append(Plan['Info']['dist'],
                                                         targetInfo['dist'])
                size = TargetDataSampler.getSize(Plan['Data'])
                Plan['msg'] = "TargetData: size %d" % (size)

            if self.isLastBatch(lapFrac) and 'Info' in Plan:
                if 'dist' in Plan['Info']:
                    dist = Plan['Info']['dist']
                    sortIDs = np.argsort(
                        dist)[:self.algParams['birth']['targetMaxSize']]
                    Plan['Data'] = Plan['Data'].select_subset_by_mask(sortIDs)
                    size = TargetDataSampler.getSize(Plan['Data'])
                    Plan['msg'] = "TargetData: size %d" % (size)
        return BirthPlans

    def birth_get_all_new_comps(self, BirthResults):
        ''' Returns list of integer ids of all new components added by
              birth moves summarized in BirthResults

        Returns
        -------
        birthCompIDs : list of int
            each entry is index of a new component
        '''
        birthCompIDs = list()
        for MoveInfo in BirthResults:
            birthCompIDs.extend(MoveInfo['birthCompIDs'])
        return birthCompIDs

    def birth_get_all_modified_comps(self, BirthResults):
        ''' Get list of int ids of all new components added by birth.

        Returns
        -------
        mCompIDs : list of integers, each entry is index of modified comp
        '''
        mCompIDs = list()
        for MoveInfo in BirthResults:
            mCompIDs.extend(MoveInfo['modifiedCompIDs'])
        return mCompIDs

    def birth_count_new_comps(self, BirthResults):
        ''' Get total number of new components added by moves.

        Returns
        -------
        Kextra : int number of components added by given list of moves
        '''
        Kextra = 0
        for MoveInfo in BirthResults:
            Kextra += len(MoveInfo['birthCompIDs'])
        return Kextra

    def birth_makeEligibilityHist(self, SS):
        targetMinSize = self.algParams['birth']['targetMinSize']
        MAX_FAIL = self.algParams['birth']['birthFailLimit']

        # Initialize histogram bins to 0
        Hist = dict(Ntoosmall=0, Ndisabled=0, Nable=0)
        for nStrike in range(MAX_FAIL):
            Hist['Nable' + str(nStrike)] = 0

        CompStatus = dict()
        for kk, compID in enumerate(self.ActiveIDVec):
            if SS.getCountVec()[kk] < targetMinSize:
                Hist['Ntoosmall'] += 1
                CompStatus[compID] = 'toosmall'
            elif compID in self.BirthRecordsByComp:
                nFail = self.BirthRecordsByComp[compID]['nFail']
                if nFail < MAX_FAIL:
                    Hist['Nable' + str(nFail)] += 1
                    Hist['Nable'] += 1
                    CompStatus[compID] = 'able-' + str(nFail)
                else:
                    Hist['Ndisabled'] += 1
                    CompStatus[compID] = 'disabled'
            else:
                Hist['Nable0'] += 1
                Hist['Nable'] += 1
                CompStatus[compID] = 'able-0'

        msg = 'Eligibility Hist:'
        for key in sorted(Hist.keys()):
            msg += " %s=%d" % (key, Hist[key])
        return Hist, CompStatus, msg

    def preparePlansForMerge(self, hmodel, SS, prevPrepInfo=None,
                             order=None,
                             BirthResults=list(),
                             lapFrac=0):

        MergeLogger.logPhase('MERGE Plans at lap ' + str(lapFrac))
        if prevPrepInfo is None:
            prevPrepInfo = dict()
        if 'PairScoreMat' not in prevPrepInfo:
            prevPrepInfo['PairScoreMat'] = None

        if SS is not None:
            # Remove any merge terms left over from previous lap
            SS.setMergeFieldsToZero()
            if SS.hasMergeTerms():
                delattr(SS, '_MergeTerms')

        mergeStartLap = self.algParams['merge']['mergeStartLap']
        mergePairSelection = self.algParams['merge']['mergePairSelection']
        mergeELBOTrackMethod = self.algParams['merge']['mergeELBOTrackMethod']
        refreshInterval = self.algParams['merge']['mergeScoreRefreshInterval']

        PrepInfo = dict()
        PrepInfo['doPrecompMergeEntropy'] = 1
        PrepInfo['mergePairSelection'] = mergePairSelection
        PrepInfo['mPairIDs'] = list()
        PrepInfo['PairScoreMat'] = None

        # Update stored ScoreMatrix to account for recent births/merges
        if hasValidKey('PairScoreMat', prevPrepInfo):
            MM = prevPrepInfo['PairScoreMat']
            # Replay any sequence-specific created states
            if self.hasMove('seqcreate'):
                Korig = MM.shape[0]
                # Expand to max size before deletes happened
                # but after merges happened
                Kmax, Kextra = self.CreateRecords[np.ceil(lapFrac - 1)]
                if Kextra > 0:
                    Kmax = Kmax - len(self.MergeLog)
                    if Korig < Kmax:
                        Mnew = np.zeros((Kmax, Kmax))
                        Mnew[:Korig, :Korig] = MM
                        MM = Mnew

            # Replay any shuffles
            if order is not None:
                Ktmp = len(order)
                Mnew = np.zeros_like(MM)
                for kA in range(Ktmp):
                    nA = np.flatnonzero(order == kA)
                    for kB in range(kA + 1, Ktmp):
                        nB = np.flatnonzero(order == kB)
                        mA = np.minimum(nA, nB)
                        mB = np.maximum(nA, nB)
                        Mnew[mA, mB] = MM[kA, kB]
                MM = Mnew

            # Replay any recent deletes
            if hasattr(self, 'DeleteAcceptRecord'):
                if 'acceptedUIDs' in self.DeleteAcceptRecord:
                    acceptedUIDs = self.DeleteAcceptRecord['acceptedUIDs']
                    origUIDs = [x for x in self.DeleteAcceptRecord['origUIDs']]
                    origUIDs = np.asarray(origUIDs)
                    for uID in acceptedUIDs:
                        kk = np.flatnonzero(origUIDs == uID)[0]
                        MM = np.delete(MM, kk, axis=0)
                        MM = np.delete(MM, kk, axis=1)
                        origUIDs = np.delete(origUIDs, kk)

            # Replay any recent birth moves!
            if len(BirthResults) > 0:
                Korig = MM.shape[0]
                Mnew = np.zeros((SS.K, SS.K))
                Mnew[:Korig, :Korig] = MM
                MM = Mnew
            assert MM.shape[0] == SS.K

            # Refresh values
            if np.floor(lapFrac) % refreshInterval == 0:
                MM.fill(0)  # Refresh!
            prevPrepInfo['PairScoreMat'] = MM

        # Determine which merge pairs we will track in the upcoming lap
        if mergePairSelection == 'wholeELBObetter':
            mPairIDs, PairScoreMat = MergePlanner.preselectPairs(
                hmodel, SS, lapFrac,
                prevScoreMat=prevPrepInfo['PairScoreMat'],
                **self.algParams['merge'])
        else:
            mPairIDs, PairScoreMat = MergePlanner.preselect_candidate_pairs(
                hmodel, SS,
                randstate=self.PRNG,
                returnScoreMatrix=1,
                M=prevPrepInfo['PairScoreMat'],
                **self.algParams['merge'])

        PrepInfo['mPairIDs'] = mPairIDs
        PrepInfo['PairScoreMat'] = PairScoreMat
        TOL = MergePlanner.ELBO_GAP_ACCEPT_TOL
        MergeLogger.log('MERGE Num pairs selected: %d/%d'
                        % (len(mPairIDs), np.sum(PairScoreMat > -1 * TOL)),
                        level='debug')

        degree = MergePlanner.calcDegreeFromEdgeList(mPairIDs, SS.K)
        if np.sum(degree > 0) > 0:
            degree = degree[degree > 0]
            MergeLogger.log('Num comps in >=1 pair: %d' %
                            (degree.size), 'debug')
            MergeLogger.log(
                'Degree distribution among selected pairs', 'debug')
            for p in [10, 50, 90, 100]:
                MergeLogger.log('   %d: %d' %
                                (p, np.percentile(degree, p)), 'debug')

        return PrepInfo

    def run_many_merge_moves(self, hmodel, SS,
                             evBound, lapFrac, MergePrepInfo):
        ''' Run (potentially many) merge moves on hmodel.

        Performing necessary bookkeeping to
        (1) avoid trying the same merge twice
        (2) avoid merging a component that has already been merged,
            since the precomputed entropy will no longer be correct.

        Returns
        -------
        hmodel : bnpy HModel, with (possibly) merged components
        SS : bnpy SuffStatBag, with (possibly) merged components
        evBound : correct ELBO for returned hmodel
                  guaranteed to be at least as large as input evBound
        '''
        MergeLogger.logPhase('MERGE Moves at lap ' + str(lapFrac))

        no_mPairIDs = 'mPairIDs' not in MergePrepInfo
        no_mPairIDs = no_mPairIDs or MergePrepInfo['mPairIDs'] is None
        if no_mPairIDs:
            MergePrepInfo['mPairIDs'] = list()

        if 'PairScoreMat' not in MergePrepInfo:
            MergePrepInfo['PairScoreMat'] = None

        Korig = SS.K
        hmodel, SS, newEvBound, Info = MergeMove.run_many_merge_moves(
            hmodel, SS, evBound,
            mPairIDs=MergePrepInfo['mPairIDs'],
            M=MergePrepInfo['PairScoreMat'],
            **self.algParams['merge'])

        # Adjust indexing for counter that determines which comp to target
        if self.hasMove('birth'):
            for kA, kB in Info['AcceptedPairs']:
                self._resetLapsSinceLastBirthAfterMerge(kA, kB)

        # Record accepted moves, so can adjust memoized stats later
        self.MergeLog = list()
        for kA, kB in Info['AcceptedPairs']:
            self.ActiveIDVec = np.delete(self.ActiveIDVec, kB, axis=0)
            self.MergeLog.append(dict(kA=kA, kB=kB, Korig=Korig))
            self.lapLastAcceptedMerge = lapFrac
            Korig -= 1

        # Reset all precalculated merge terms
        if SS.hasMergeTerms():
            SS.setMergeFieldsToZero()

        # ScoreMat here will have shape Ka x Ka, where Ka <= K
        # Ka < K in the case of batch-specific births
        # whose new comps aren't tracked)
        # ScoreMat will be updated to size SS.K,SS.K in preparePlansForMerge()
        MergePrepInfo['PairScoreMat'] = Info['ScoreMat']
        MergePrepInfo['mPairIDs'] = list()
        return hmodel, SS, newEvBound

    def _resetLapsSinceLastBirthAfterMerge(self, kA, kB):
        ''' Update self.LapsSinceLastBirth to reflect accepted merge

        Post Condition
        ---------
        None. Updates to self.LapsSinceLastBirth happen in-place.
        '''
        compList = list(self.LapsSinceLastBirth.keys())
        newDict = defaultdict(int)
        for kk in compList:
            if kk == kA:
                newDict[kA] = np.maximum(self.LapsSinceLastBirth[kA],
                                         self.LapsSinceLastBirth[kB])
            elif kk < kB:
                newDict[kk] = self.LapsSinceLastBirth[kk]
            elif kk > kB:
                newDict[kk - 1] = self.LapsSinceLastBirth[kk]
        self.LapsSinceLastBirth = newDict

    def doDeleteAtLap(self, lapFrac):
        if 'delete' not in self.algParams:
            return False
        return lapFrac >= self.algParams['delete']['deleteStartLap']

    def deleteCollectTarget(self, Dchunk, hmodel, LPchunk,
                            batchID,
                            DeletePlans):
        """ Add relevant subset of data from provided chunk to Plan

            Returns
            -------
            DeletePlans. Updated in place.
        """
        for planID, Plan in enumerate(DeletePlans):
            Plan = DCollector.addDataFromBatchToPlan(
                Plan, hmodel, Dchunk, LPchunk,
                batchID=batchID,
                uIDs=self.ActiveIDVec,
                maxUID=self.maxUID,
                lapFrac=self.lapFrac,
                isFirstBatch=self.isFirstBatch(self.lapFrac),
                isLastBatch=self.isLastBatch(self.lapFrac),
                **self.algParams['delete'])
            if len(list(Plan.keys())) == 0:
                # Empty Plan dict means it went over budget
                # So, we should remove this Plan from consideration
                DeletePlans.pop(planID)
        return DeletePlans

    def deleteAndUpdateMemory(self, hmodel, SS, DeletePlans):
        """ Construct and evaluate delete proposals.

            Returns
            -------
            hmodel : bnpy.HModel with updated fields
            SS : bnpy.suffstats.SuffStatBag, with updated fields
        """
        self.ELBOReady = True
        self.DeleteAcceptRecord = dict()
        if self.lapFrac <= 1 or SS is None:
            return hmodel, SS

        # Make last minute plan for any empty comps
        EPlan = DPlanner.makePlanForEmptyComps(SS, **self.algParams['delete'])
        if hasattr(self, 'MergeLog') and len(self.MergeLog) > 0:
            # Accepted merge means skip all deletes except the trivial ones
            DeleteLogger.log('ABANDONED planned delete due to accepted merge.')
            if 'candidateUIDs' in EPlan:
                DeletePlans = [EPlan]
            else:
                return hmodel, SS
        else:
            if 'candidateUIDs' in EPlan:
                nEmpty = len(EPlan['candidateUIDs'])
                DeleteLogger.log('Last-minute Plan: %d empty' % (nEmpty))

                # Adjust the existing plan so comps deleted by EmptyPlan
                # are not repeated later
                if len(DeletePlans) > 0:
                    DPlan = DeletePlans[0]
                    remIDs = list()
                    for ii, uid in enumerate(DPlan['candidateUIDs']):
                        if uid in EPlan['candidateUIDs']:
                            remIDs.append(ii)
                    for ii in reversed(sorted(remIDs)):
                        DPlan['candidateUIDs'].pop(ii)

                    if len(DPlan['candidateUIDs']) == 0:
                        DeletePlans = [EPlan]
                    else:
                        DeletePlans = [DPlan, EPlan]
                else:
                    DeletePlans = [EPlan]

        # Evaluate each proposal
        newSS = SS.copy()
        newModel = hmodel.copy()
        for moveID, DPlan in enumerate(DeletePlans):
            if 'DTargetData' in DPlan:
                # Updates SSmemory in-place
                newModel, newSS, self.SSmemory, DPlan = \
                    DEvaluator.runDeleteMoveAndUpdateMemory(
                        newModel, newSS, DPlan,
                        LPkwargs=self.algParamsLP,
                        SSmemory=self.SSmemory,
                        lapFrac=self.lapFrac,
                        **self.algParams['delete'])
                nYes = len(DPlan['acceptedUIDs'])
                nAttempt = len(DPlan['candidateUIDs'])
                DeleteLogger.log('DELETE %d/%d accepted' % (nYes, nAttempt),
                                 'info')
            else:
                # Auto-accepted delete (specific only for empty comps)
                DPlan['didAccept'] = 2
                DPlan['acceptedUIDs'] = DPlan['candidateUIDs']

                # Make all stats stored in Memory have correct size
                for uID in DPlan['acceptedUIDs']:
                    kk = np.flatnonzero(newSS.uIDs == uID)[0]
                    newSS.removeComp(kk)
                    for batchID in self.SSmemory:
                        if kk < self.SSmemory[batchID].K:
                            if self.SSmemory[batchID].uIDs[kk] == uID:
                                self.SSmemory[batchID].removeComp(kk)

                # Reset all ELBO and Merge terms stored in Memory
                for batchID in self.SSmemory:
                    self.SSmemory[batchID].setELBOFieldsToZero()
                    self.SSmemory[batchID].setMergeFieldsToZero()

                newSS.setELBOFieldsToZero()
                newSS.setMergeFieldsToZero()
                newModel.update_global_params(newSS)
                # Do extra update, to improve numerical
                # values found by gradient descent
                newModel.allocModel.update_global_params(newSS)
                newModel.allocModel.update_global_params(newSS)

                nEmpty = len(DPlan['candidateUIDs'])
                DeleteLogger.log('DELETED %d empty comps' % (nEmpty),
                                 'info')

            # -------------------    Update DeleteRecords
            for uID in DPlan['candidateUIDs']:
                if uID in DPlan['acceptedUIDs']:
                    if uID in self.DeleteRecordsByComp:
                        del self.DeleteRecordsByComp[uID]
                else:
                    if uID not in self.DeleteRecordsByComp:
                        self.DeleteRecordsByComp[uID]['nFail'] = 0
                    self.DeleteRecordsByComp[uID]['nFail'] += 1

            if DPlan['didAccept']:
                self.ELBOReady = False
                self.ActiveIDVec = newSS.uIDs.copy()
                self.lapLastAcceptedDelete = self.lapFrac

                acceptedUIDs = [x for x in DPlan['acceptedUIDs']]
                if 'origUIDs' not in self.DeleteAcceptRecord:
                    self.DeleteAcceptRecord['origUIDs'] = SS.uIDs.copy()
                    self.DeleteAcceptRecord['acceptedUIDs'] = acceptedUIDs
                else:
                    self.DeleteAcceptRecord[
                        'acceptedUIDs'].extend(acceptedUIDs)
            for batchID in self.SSmemory:
                assert np.allclose(
                    self.SSmemory[batchID].uIDs, self.ActiveIDVec)
        # <<< end for loop over DeletePlans

        # Verify post-condition: same states represented by newSS and SSmemory
        for batchID in self.SSmemory:
            assert newSS.K == self.SSmemory[batchID].K
            assert np.allclose(newSS.uIDs, self.SSmemory[batchID].uIDs)

            if newSS.K < SS.K:
                for key in list(self.SSmemory[batchID]._ELBOTerms._FieldDims.keys()):
                    arr = self.SSmemory[batchID].getELBOTerm(key)
                    assert np.allclose(0, arr)
        if newSS.K < SS.K:
            assert self.ELBOReady is False
            for key in list(newSS._ELBOTerms._FieldDims.keys()):
                arr = newSS.getELBOTerm(key)
                assert np.allclose(0, arr)
        else:
            assert self.ELBOReady is True

        # TODO adjust LPmemory??
        return newModel, newSS

    def verifyELBOTracking(self, hmodel, SS, evBound=None,
                           BirthResults=list(),
                           **kwargs):
        ''' Verify current aggregated SS consistent with sum over all batches
        '''
        if self.doDebugVerbose():
            self.print_msg(
                '>>>>>>>> BEGIN double-check @ lap %.2f' % (self.lapFrac))

        if evBound is None:
            evBound = hmodel.calc_evidence(SS=SS)

        # All merges and deletes should be fast forwarded anyway
        tmpLog = self.MergeLog
        self.MergeLog = []
        # Reconstruct aggregate SS explicitly by sum over all stored batches
        for batchID in range(len(list(self.SSmemory.keys()))):
            SSchunk = self.load_batch_suff_stat_from_memory(
                batchID, doCopy=1, order=None, Kfinal=SS.K)
            if batchID == 0:
                SS2 = SSchunk.copy()
            else:
                SS2 += SSchunk
        self.MergeLog = tmpLog

        # Add in extra mass from birth moves
        for MoveInfo in BirthResults:
            if MoveInfo['didAddNew'] and 'extraSS' in MoveInfo:
                if 'extraSSDone' not in MoveInfo:
                    extraSS = MoveInfo['extraSS'].copy()
                    if extraSS.K < SS2.K:
                        extraSS.insertEmptyComps(SS2.K - extraSS.K)
                    SS2 += extraSS

        evCheck = hmodel.calc_evidence(SS=SS2)
        if self.doDebugVerbose():
            self.print_msg('% 14.8f evBound from agg SS' % (evBound))
            self.print_msg(
                '% 14.8f evBound from sum over SSmemory' % (evCheck))

            if not self.ELBOReady:
                self.print_msg('    ELBO not ready. Disregard mismatch here.')

        condCount = np.allclose(SS.getCountVec(), SS2.getCountVec())
        condELBO = np.allclose(evBound, evCheck) or not self.ELBOReady
        condUIDs = np.allclose(SS.uIDs, SS2.uIDs)

        if self.algParams['debug'].count('interactive'):
            isCorrect = condCount and condUIDs and condELBO
            if not isCorrect:
                from IPython import embed
                embed()
        else:
            assert condELBO
            assert condCount
            assert condUIDs

        if self.doDebugVerbose():
            self.print_msg(
                '<<<<<<<< END   double-check @ lap %.2f' % (self.lapFrac))

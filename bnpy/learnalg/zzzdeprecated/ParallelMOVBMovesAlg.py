'''
ParallelVBAlg.py

Implementation of parallel memoized variational algorithm for bnpy models.
'''
import numpy as np
import multiprocessing

from LearnAlg import makeDictOfAllWorkspaceVars
from MOVBBirthMergeAlg import MOVBBirthMergeAlg
from bnpy.util import sharedMemDictToNumpy, sharedMemToNumpyArray
from SharedMemWorker import SharedMemWorker


class ParallelMOVBMovesAlg(MOVBBirthMergeAlg):

    def __init__(self, **kwargs):
        ''' Constructor for ParallelMOVBAlg
        '''
        # Initialize instance vars related to
        # birth / merge / delete records
        MOVBBirthMergeAlg.__init__(self, **kwargs)

        self.nWorkers = self.algParams['nWorkers']
        maxWorkers = multiprocessing.cpu_count()
        if self.nWorkers < 0:
            self.nWorkers = maxWorkers + self.nWorkers
        if self.nWorkers > maxWorkers:
            self.nWorkers = np.maximum(self.nWorkers, maxWorkers)
        self.memoLPkeys = []

    def fit(self, hmodel, DataIterator, LP=None, **kwargs):
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

        # Initialize Progress Tracking vars like nBatch, lapFrac, etc.
        iterid, lapFrac = self.initProgressTrackVars(DataIterator)

        # Save initial state
        self.saveParams(lapFrac, hmodel)

        # Custom func hook
        self.eval_custom_func(
            isInitial=1, **makeDictOfAllWorkspaceVars(**vars()))

        # Setup workers for parallel runs
        if self.nWorkers > 0:
            JobQ, ResultQ, aSharedMem, oSharedMem = setupQueuesAndWorkers(
                DataIterator, hmodel,
                nWorkers=self.nWorkers,
                algParamsLP=self.algParamsLP)
            self.JobQ = JobQ
            self.ResultQ = ResultQ

        # Prep for birth
        BirthPlans = list()
        BirthResults = list()
        prevBirthResults = list()

        # Prep for merge
        MergePrepInfo = dict()
        if self.hasMove('merge'):
            mergeStartLap = self.algParams['merge']['mergeStartLap']
        else:
            mergeStartLap = 0
        order = None

        # Prep for delete
        DeletePlans = list()

        # Begin loop over batches of data...
        SS = None
        isConverged = False
        self.set_start_time_now()
        while DataIterator.has_next_batch():

            batchID = DataIterator.get_next_batch(batchIDOnly=1)

            # Update progress-tracking variables
            iterid += 1
            lapFrac = (iterid + 1) * self.lapFracInc
            self.lapFrac = lapFrac
            self.set_random_seed_at_lap(lapFrac)
            if self.doDebugVerbose():
                self.print_msg('========================== lap %.2f batch %d'
                               % (lapFrac, batchID))

            # Prepare for merges
            if self.hasMove('merge') and self.doMergePrepAtLap(lapFrac):
                MergePrepInfo = self.preparePlansForMerge(
                    hmodel, SS, MergePrepInfo,
                    order=order,
                    BirthResults=BirthResults,
                    lapFrac=lapFrac)
            elif self.isFirstBatch(lapFrac):
                if self.doMergePrepAtLap(lapFrac + 1):
                    MergePrepInfo = dict(
                        mergePairSelection=self.algParams[
                            'merge']['mergePairSelection'])
                else:
                    MergePrepInfo = dict()

            # Reset selection terms to zero
            if self.isFirstBatch(lapFrac):
                if SS is not None and SS.hasSelectionTerms():
                    SS._SelectTerms.setAllFieldsToZero()

            # Update shared memory with new global params
            if self.nWorkers > 0:
                aSharedMem = hmodel.allocModel.fillSharedMemDictForLocalStep(
                    aSharedMem)
                oSharedMem = hmodel.obsModel.fillSharedMemDictForLocalStep(
                    oSharedMem)

            # Local/Summary step for current batch
            self.algParamsLP['lapFrac'] = lapFrac  # for logging
            self.algParamsLP['batchID'] = batchID
            if self.nWorkers > 0:
                SSchunk = self.calcLocalParamsAndSummarize_parallel(
                    DataIterator, hmodel,
                    MergePrepInfo=MergePrepInfo,
                    batchID=batchID, lapFrac=lapFrac)
            else:
                SSchunk = self.calcLocalParamsAndSummarize_main(
                    DataIterator, hmodel,
                    MergePrepInfo=MergePrepInfo,
                    batchID=batchID, lapFrac=lapFrac)

            self.saveDebugStateAtBatch(
                'Estep', batchID, SSchunk=SSchunk, SS=SS, hmodel=hmodel)

            # Summary step for whole-dataset stats
            # (does incremental update)
            SS = self.memoizedSummaryStep(hmodel, SS, SSchunk, batchID)

            # Global step
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
                        DPlan['targetSS'].reorderComps(order)
                        targetSSbyBatch = DPlan['targetSSByBatch']
                        for batchID in targetSSbyBatch:
                            targetSSbyBatch[batchID].reorderComps(order)
                    # Update memoized stats for each batch
                    self.fastForwardMemory(Kfinal=SS.K, order=order)

            # ELBO calculation
            nLapsCompleted = lapFrac - self.algParams['startLap']
            if nLapsCompleted > 1.0:
                # evBound increases monotonically AFTER first lap
                # verify_evidence warns if this isn't happening
                self.verify_evidence(evBound, prevBound, lapFrac)

            if self.doDebug() and lapFrac >= 1.0:
                self.verifyELBOTracking(hmodel, SS, evBound, order=order)

            self.saveDebugStateAtBatch(
                'Mstep', batchID, SSchunk=SSchunk, SS=SS, hmodel=hmodel)

            # Assess convergence
            countVec = SS.getCountVec()
            if lapFrac > 1.0:
                isConverged = self.isCountVecConverged(countVec, prevCountVec)
                hasMoreMoves = self.hasMoreReasonableMoves(lapFrac, SS)
                isConverged = isConverged and not hasMoreMoves
                self.setStatus(lapFrac, isConverged)

            # Display progress
            if self.isLogCheckpoint(lapFrac, iterid):
                self.printStateToLog(hmodel, evBound, lapFrac, iterid)

            # Save diagnostics and params
            if self.isSaveDiagnosticsCheckpoint(lapFrac, iterid):
                self.saveDiagnostics(lapFrac, SS, evBound)
            if self.isSaveParamsCheckpoint(lapFrac, iterid):
                self.saveParams(lapFrac, hmodel, SS)

            # Custom func hook
            self.eval_custom_func(**makeDictOfAllWorkspaceVars(**vars()))

            if isConverged and self.isLastBatch(lapFrac) \
               and nLapsCompleted >= self.algParams['minLaps']:
                break
            prevCountVec = countVec.copy()
            prevBound = evBound
            # .... end loop over data

        # Finished! Save, print and exit
        for workerID in range(self.nWorkers):
            # Passing None to JobQ is shutdown signal
            self.JobQ.put(None)

        self.printStateToLog(hmodel, evBound, lapFrac, iterid, isFinal=1)
        self.saveParams(lapFrac, hmodel, SS)
        self.eval_custom_func(
            isFinal=1, **makeDictOfAllWorkspaceVars(**vars()))

        # Births and merges require copies of original model object
        #  we need to make sure original reference has updated parameters, etc.
        if id(origmodel) != id(hmodel):
            origmodel.allocModel = hmodel.allocModel
            origmodel.obsModel = hmodel.obsModel

        # Return information about this run
        return self.buildRunInfo(evBound=evBound, SS=SS,
                                 SSmemory=self.SSmemory)

    def memoizedSummaryStep(self, hmodel, SS, SSchunk, batchID,
                            MergePrepInfo=None,
                            order=None,
                            **kwargs):
        ''' Execute summary step on current batch and update aggregated SS.

        Returns
        --------
        SS : updated aggregate suff stats
        '''
        if batchID in self.SSmemory:
            oldSSchunk = self.load_batch_suff_stat_from_memory(
                batchID, doCopy=0, Kfinal=SS.K, order=order)
            assert not oldSSchunk.hasMergeTerms()
            assert oldSSchunk.K == SS.K
            assert np.allclose(SS.uIDs, oldSSchunk.uIDs)
            SS -= oldSSchunk

        # UIDs are not set by parallel workers. Need to do this here
        SSchunk.setUIDs(self.ActiveIDVec.copy())
        if SS is None:
            SS = SSchunk.copy()
        else:
            assert SSchunk.K == SS.K
            assert np.allclose(SSchunk.uIDs, self.ActiveIDVec)
            assert np.allclose(SS.uIDs, self.ActiveIDVec)
            SS += SSchunk
            if not SS.hasSelectionTerms() and SSchunk.hasSelectionTerms():
                SS._SelectTerms = SSchunk._SelectTerms
        assert hasattr(SS, 'uIDs')
        self.save_batch_suff_stat_to_memory(batchID, SSchunk)

        # Force aggregated suff stats to obey required constraints.
        # This avoids numerical issues caused by incremental updates
        if hasattr(hmodel.allocModel, 'forceSSInBounds'):
            hmodel.allocModel.forceSSInBounds(SS)
        if hasattr(hmodel.obsModel, 'forceSSInBounds'):
            hmodel.obsModel.forceSSInBounds(SS)
        return SS

    def calcLocalParamsAndSummarize_main(self,
                                         DataIterator, hmodel,
                                         MergePrepInfo=None,
                                         batchID=0, **kwargs):
        ''' Execute local step and summary step in main process.

        Returns
        -------
        SSagg : bnpy.suffstats.SuffStatBag
            Aggregated suff stats from all processed slices of the data.
        '''
        if not isinstance(MergePrepInfo, dict):
            MergePrepInfo = dict()
        LPkwargs = self.algParamsLP
        LPkwargs.update(MergePrepInfo)

        Dbatch = DataIterator.getBatch(batchID=batchID)
        LPbatch = hmodel.calc_local_params(Dbatch, **LPkwargs)
        SSbatch = hmodel.get_global_suff_stats(
            Dbatch, LPbatch, doPrecompEntropy=1, **MergePrepInfo)
        SSbatch.setUIDs(self.ActiveIDVec.copy())
        return SSbatch

    def calcLocalParamsAndSummarize_parallel(self,
                                             DataIterator, hmodel,
                                             MergePrepInfo=None,
                                             batchID=0, lapFrac=-1, **kwargs):
        ''' Execute local step and summary step in parallel via workers.

        Returns
        -------
        SSagg : bnpy.suffstats.SuffStatBag
            Aggregated suff stats from all processed slices of the data.
        '''
        # Map Step
        # Create several tasks (one per worker) and add to job queue
        nWorkers = self.algParams['nWorkers']
        for workerID in range(nWorkers):
            sliceArgs = DataIterator.calcSliceArgs(
                batchID, workerID, nWorkers, lapFrac)
            aArgs = hmodel.allocModel.getSerializableParamsForLocalStep()
            aArgs.update(MergePrepInfo)
            oArgs = hmodel.obsModel.getSerializableParamsForLocalStep()
            self.JobQ.put((sliceArgs, aArgs, oArgs))

        # Pause at this line until all jobs are marked complete.
        self.JobQ.join()

        # Reduce step
        # Aggregate results across across all workers
        SSagg = self.ResultQ.get()
        while not self.ResultQ.empty():
            SSslice = self.ResultQ.get()
            SSagg += SSslice
        return SSagg


def setupQueuesAndWorkers(DataIterator, hmodel,
                          algParamsLP=None,
                          nWorkers=0,
                          **kwargs):
    ''' Create pool of worker processes for provided dataset and model.

    Returns
    -------
    JobQ : multiprocessing task queue
        Used for passing tasks to workers
    ResultQ : multiprocessing task Queue
        Used for receiving SuffStatBags from workers
    '''
    # Create a JobQ (to hold tasks to be done)
    # and a ResultsQ (to hold results of completed tasks)
    manager = multiprocessing.Manager()
    JobQ = manager.Queue()
    ResultQ = manager.Queue()

    # Get the function handles
    makeDataSliceFromSharedMem = DataIterator.getDataSliceFunctionHandle()
    o_calcLocalParams, o_calcSummaryStats = hmodel.obsModel.\
        getLocalAndSummaryFunctionHandles()
    a_calcLocalParams, a_calcSummaryStats = hmodel.allocModel.\
        getLocalAndSummaryFunctionHandles()

    # Create the shared memory
    try:
        dataSharedMem = DataIterator.getRawDataAsSharedMemDict()
    except AttributeError as e:
        dataSharedMem = None
    aSharedMem = hmodel.allocModel.fillSharedMemDictForLocalStep()
    oSharedMem = hmodel.obsModel.fillSharedMemDictForLocalStep()

    # Create multiple workers
    for uid in range(nWorkers):
        worker = SharedMemWorker(uid, JobQ, ResultQ,
                                 makeDataSliceFromSharedMem,
                                 o_calcLocalParams,
                                 o_calcSummaryStats,
                                 a_calcLocalParams,
                                 a_calcSummaryStats,
                                 dataSharedMem,
                                 aSharedMem,
                                 oSharedMem,
                                 LPkwargs=algParamsLP,
                                 verbose=1)
        worker.start()

    return JobQ, ResultQ, aSharedMem, oSharedMem

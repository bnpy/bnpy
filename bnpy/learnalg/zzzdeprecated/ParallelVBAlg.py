'''
ParallelVBAlg.py

Implementation of parallel variational bayes algorithm for bnpy models.
'''
import numpy as np
import multiprocessing

from LearnAlg import LearnAlg, makeDictOfAllWorkspaceVars
from bnpy.util import sharedMemDictToNumpy, sharedMemToNumpyArray
from SharedMemWorker import SharedMemWorker


class ParallelVBAlg(LearnAlg):

    def __init__(self, **kwargs):
        ''' Create VBLearnAlg, subtype of generic LearnAlg
        '''
        LearnAlg.__init__(self, **kwargs)
        self.nWorkers = self.algParams['nWorkers']
        # if not specified, break up into number of parallel ones
        if self.nWorkers == 0:
            self.nWorkers = multiprocessing.cpu_count()

    def fit(self, hmodel, Data, LP=None):
        ''' Run VB learning algorithm, fit global parameters of hmodel to Data
            Returns
            --------
            Info : dict of run information, with fields
            * evBound : final ELBO evidence bound
            * status : str message indicating reason for termination
                       {'converged', 'max laps exceeded'}
            * LP : dict of local parameters for final model
        '''
        prevBound = -np.inf
        isConverged = False

        # Save initial state
        self.saveParams(0, hmodel)

        # Custom func hook
        self.eval_custom_func(
            isInitial=1, **makeDictOfAllWorkspaceVars(**vars()))

        self.set_start_time_now()

        # TODO: delete this, this is simply for debugging purposes
        isParallel = True
        self.nDoc = Data.nDoc
        if isParallel:
            # Create a JobQ (to hold tasks to be done)
            # and a ResultsQ (to hold results of completed tasks)
            manager = multiprocessing.Manager()
            self.JobQ = manager.Queue()
            self.ResultQ = manager.Queue()

            # Get the function handles
            makeDataSliceFromSharedMem = Data.getDataSliceFunctionHandle()
            o_calcLocalParams, o_calcSummaryStats = hmodel.obsModel.\
                getLocalAndSummaryFunctionHandles()
            a_calcLocalParams, a_calcSummaryStats = hmodel.allocModel.\
                getLocalAndSummaryFunctionHandles()

            # Create the shared memory
            dataSharedMem = Data.getRawDataAsSharedMemDict()
            aSharedMem = hmodel.allocModel.fillSharedMemDictForLocalStep()
            oSharedMem = hmodel.obsModel.fillSharedMemDictForLocalStep()

            # Create multiple workers
            for uid in range(self.nWorkers):
                SharedMemWorker(uid, self.JobQ, self.ResultQ,
                                makeDataSliceFromSharedMem,
                                o_calcLocalParams,
                                o_calcSummaryStats,
                                a_calcLocalParams,
                                a_calcSummaryStats,
                                dataSharedMem,
                                aSharedMem,
                                oSharedMem,
                                LPkwargs=self.algParamsLP,
                                verbose=1).start()
        else:
            # Need to store shared mem

            aSharedMem = hmodel.allocModel.fillSharedMemDictForLocalStep()
            oSharedMem = hmodel.obsModel.fillSharedMemDictForLocalStep()
            self.dataSharedMem = Data.getRawDataAsSharedMemDict()
            self.makeDataSliceFromSharedMem = Data.getDataSliceFunctionHandle()

        for iterid in range(1, self.algParams['nLap'] + 1):
            lap = self.algParams['startLap'] + iterid
            nLapsCompleted = lap - self.algParams['startLap']
            self.set_random_seed_at_lap(lap)

            if isParallel:
                SS = self.calcLocalParamsAndSummarize(
                    hmodel)  # TODO fill in params

            else:
                SS = self.serialCalcLocalParamsAndSummarize(hmodel)

            # Global/M step
            hmodel.update_global_params(SS)

            # update the memory
            aSharedMem = hmodel.allocModel.fillSharedMemDictForLocalStep(
                aSharedMem)
            oSharedMem = hmodel.obsModel.fillSharedMemDictForLocalStep(
                oSharedMem)

            # ELBO calculation
            evBound = hmodel.calc_evidence(Data=Data, SS=SS)

            if lap > 1.0:
                # Report warning if bound isn't increasing monotonically
                self.verify_evidence(evBound, prevBound)

            # Check convergence of expected counts
            countVec = SS.getCountVec()
            if lap > 1.0:
                isConverged = self.isCountVecConverged(countVec, prevCountVec)
                self.setStatus(lap, isConverged)

            # Display progress
            self.updateNumDataProcessed(Data.get_size())
            if self.isLogCheckpoint(lap, iterid):
                self.printStateToLog(hmodel, evBound, lap, iterid)

            # Save diagnostics and params
            if self.isSaveDiagnosticsCheckpoint(lap, iterid):
                self.saveDiagnostics(lap, SS, evBound)
            if self.isSaveParamsCheckpoint(lap, iterid):
                self.saveParams(lap, hmodel, SS)

            # Custom func hook
            self.eval_custom_func(**makeDictOfAllWorkspaceVars(**vars()))

            if nLapsCompleted >= self.algParams['minLaps'] and isConverged:
                break
            prevBound = evBound
            prevCountVec = countVec.copy()
            # .... end loop over laps

        # Finished! Save, print and exit
        for workerID in range(self.nWorkers):
            # Passing None to JobQ is shutdown signal
            self.JobQ.put(None)
        self.saveParams(lap, hmodel, SS)
        self.printStateToLog(hmodel, evBound, lap, iterid, isFinal=1)
        self.eval_custom_func(
            isFinal=1, **makeDictOfAllWorkspaceVars(**vars()))

        return self.buildRunInfo(evBound=evBound, SS=SS)

    def calcLocalParamsAndSummarize(self, hmodel):
        # MAP!
        # Create several tasks (one per worker) and add to job queue
        for dataBatchID, start, stop in self.sliceGenerator(
                self.nDoc, self.nWorkers):
            sliceArgs = (dataBatchID, start, stop)
            aArgs = hmodel.allocModel.getSerializableParamsForLocalStep()
            oArgs = hmodel.obsModel.getSerializableParamsForLocalStep()
            self.JobQ.put((sliceArgs, aArgs, oArgs))

        # Pause at this line until all jobs are marked complete.
        self.JobQ.join()

        # REDUCE!
        # Aggregate results across across all workers
        SS = self.ResultQ.get()
        while not self.ResultQ.empty():
            SSchunk = self.ResultQ.get()
            SS += SSchunk
        return SS

    def serialCalcLocalParamsAndSummarize(self, hmodel):
        SSagg = None

        for dataBatchID, start, stop in self.sliceGenerator(
                self.nDoc, self.nWorkers):
            sliceArgs = (dataBatchID, start, stop)
            aArgs = hmodel.allocModel.getSerializableParamsForLocalStep()
            oArgs = hmodel.obsModel.getSerializableParamsForLocalStep()

            # Get the function handles
            o_calcLocalParams, o_calcSummaryStats = hmodel.obsModel.\
                getLocalAndSummaryFunctionHandles()
            a_calcLocalParams, a_calcSummaryStats = hmodel.allocModel.\
                getLocalAndSummaryFunctionHandles()

            # Create the shared memory

            aSharedMem = hmodel.allocModel.fillSharedMemDictForLocalStep()
            oSharedMem = hmodel.obsModel.fillSharedMemDictForLocalStep()

            cslice = (start, stop)
            Dslice = self.makeDataSliceFromSharedMem(
                self.dataSharedMem, cslice=cslice)
            aArgs.update(sharedMemDictToNumpy(aSharedMem))
            oArgs.update(sharedMemDictToNumpy(oSharedMem))

            LP = o_calcLocalParams(Dslice, **oArgs)
            LP = a_calcLocalParams(Dslice, LP, **aArgs)

            SSslice = a_calcSummaryStats(
                Dslice,
                LP,
                doPrecompEntropy=1,
                **aArgs)
            SSslice = o_calcSummaryStats(Dslice, SSslice, LP, **oArgs)
            if start == 0:
                SSagg = SSslice
            else:
                SSagg += SSslice

        return SSagg

    def sliceGenerator(self, nDoc=0, nWorkers=0):
        """ Iterate over slices given problem size and num workers

        Yields
        --------
        (start,stop) : tuple
        """
        batchSize = int(np.floor(nDoc / nWorkers))
        for workerID in range(nWorkers):
            start = workerID * batchSize
            stop = (workerID + 1) * batchSize
            if workerID == nWorkers - 1:
                stop = nDoc
            # yields batchID, start, and stop
            # For VB, batchID=0 since it is online
            yield 0, start, stop

import multiprocessing
import numpy as np
import time
import bnpy

from bnpy.util.ParallelUtil import sharedMemDictToNumpy
from RunBenchmark import sliceGenerator

def calcLocalParamsAndSummarize(
        JobQ, ResultQ, Data, hmodel, nWorker=0,
        LPkwargs=dict(),
        **kwargs):
    """ Execute  processed by workers in parallel.
    """

    LPkwargs.update(
        hmodel.obsModel.getSerializableParamsForLocalStep())
    LPkwargs.update(
        hmodel.allocModel.getSerializableParamsForLocalStep())

    # MAP step
    # Create several tasks (one per worker) and add to job queue
    for start, stop in sliceGenerator(Data, nWorker):
        JobQ.put((start, stop, LPkwargs))

    # Pause at this line until all jobs are marked complete.
    JobQ.join()

    # REDUCE step
    # Aggregate results across across all workers
    SS, telapsed_max = ResultQ.get()
    while not ResultQ.empty():
        SSslice, telapsed_cur = ResultQ.get()
        SS += SSslice
        telapsed_max = np.maximum(telapsed_max, telapsed_cur)
    return SS, telapsed_max

def setUpWorkers(
        Data=None, hmodel=None,
        nWorker=1, verbose=0, nRepsForMinDuration=1, **kwargs):
    ''' Create queues and launch all workers.

    Returns
    -------
    JobQ
    ResultQ
    '''
    # Create a JobQ (to hold tasks to be done)
    # and a ResultsQ (to hold results of completed tasks)
    manager = multiprocessing.Manager()
    JobQ = manager.Queue()
    ResultQ = manager.Queue()

    # Create sharedmem representations of Data and hmodel
    dataSharedMem = Data.getRawDataAsSharedMemDict()
    aSharedMem = hmodel.allocModel.fillSharedMemDictForLocalStep()
    oSharedMem = hmodel.obsModel.fillSharedMemDictForLocalStep()
    ShMem = dict(dataSharedMem=dataSharedMem,
                 aSharedMem=aSharedMem,
                 oSharedMem=oSharedMem)

    # Get relevant function handles
    afuncHTuple = hmodel.allocModel.getLocalAndSummaryFunctionHandles()
    ofuncHTuple = hmodel.obsModel.getLocalAndSummaryFunctionHandles()
    funcH = dict(
        makeDataSliceFromSharedMem=Data.getDataSliceFunctionHandle(),
        a_calcLocalParams=afuncHTuple[0],
        a_calcSummaryStats=afuncHTuple[1],
        o_calcLocalParams=ofuncHTuple[0],
        o_calcSummaryStats=ofuncHTuple[1],
        )

    # Launch desired number of worker processes
    # We don't need to store references to these processes,
    # We can get everything we need from JobQ and ResultsQ
    for uid in range(nWorker):
        workerProcess = Worker_SHMData_SHMModel(
            uid, JobQ, ResultQ,
            ShMem=ShMem,
            funcH=funcH,
            nReps=nRepsForMinDuration,
            verbose=verbose)
        workerProcess.start()
    return JobQ, ResultQ

def tearDownWorkers(JobQ=None, ResultQ=None, nWorker=1, **kwargs):
    ''' Shutdown pool of workers.
    '''
    for workerID in range(nWorker):
        # Passing None to JobQ is shutdown signal
        JobQ.put(None) 
    time.sleep(0.1)  # let workers all shut down before we quit


class Worker_SHMData_SHMModel(multiprocessing.Process):

    ''' Single "worker" process that processes tasks delivered via queues.

    Attributes
    ----------
    JobQ : multiprocessing.Queue
    ResultQ : multiprocessing.Queue
    '''

    def __init__(self, uid, JobQ, ResultQ, 
            ShMem=dict(),
            funcH=dict(),
            verbose=0, nReps=1):
        ''' Create single worker process, linked to provided queues.
        '''
        super(type(self), self).__init__()  # Required super constructor call
        self.uid = uid
        self.ShMem = ShMem
        self.funcH = funcH
        self.JobQ = JobQ
        self.ResultQ = ResultQ
        self.verbose = verbose
        self.nReps = nReps


    def run(self):
        ''' Perform calcLocalParamsAndSummarize on jobs in JobQ.
        
        Post Condition
        --------------
        '''

        # Construct iterator with sentinel value of None (for termination)
        jobIterator = iter(self.JobQ.get, None)

        for jobArgs in jobIterator:
            start, stop, LPkwargs = jobArgs

            Dslice = self.funcH['makeDataSliceFromSharedMem'](
                self.ShMem['dataSharedMem'], cslice=(start, stop))

            # Fill in params needed for local step
            LPkwargs.update(
                sharedMemDictToNumpy(self.ShMem['aSharedMem']))
            LPkwargs.update(
                sharedMemDictToNumpy(self.ShMem['oSharedMem']))

            tstart = time.time()
            for rep in range(self.nReps):
                # Do local step
                LP = self.funcH['o_calcLocalParams'](Dslice, **LPkwargs)
                LP = self.funcH['a_calcLocalParams'](Dslice, LP, **LPkwargs)

                # Do summary step
                SSslice = self.funcH['a_calcSummaryStats'](
                    Dslice, LP, **LPkwargs)
                SSslice = self.funcH['o_calcSummaryStats'](
                    Dslice, SSslice, LP, **LPkwargs)
            twork = time.time() - tstart

            self.ResultQ.put((SSslice, twork))
            self.JobQ.task_done()

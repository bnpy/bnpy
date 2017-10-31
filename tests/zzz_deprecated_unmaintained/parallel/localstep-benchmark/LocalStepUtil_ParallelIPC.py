import multiprocessing
import numpy as np
import time
import bnpy

from RunBenchmark import sliceGenerator

def calcLocalParamsAndSummarize(
        JobQ, ResultQ, Data, hmodel, nWorker=0,
        LPkwargs=dict(),
        **kwargs):
    """ Execute  processed by workers in parallel.
    """

    # MAP step
    # Create several tasks (one per worker) and add to job queue
    for start, stop in sliceGenerator(Data, nWorker):
        Dslice = Data.select_subset_by_mask(
            np.arange(start,stop, dtype=np.int32))
        JobQ.put((Dslice, hmodel, LPkwargs))

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

def setUpWorkers(nWorker=1, verbose=0, nRepsForMinDuration=1, **kwargs):
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

    # Launch desired number of worker processes
    # We don't need to store references to these processes,
    # We can get everything we need from JobQ and ResultsQ
    for uid in range(nWorker):
        workerProcess = Worker_IPCData_IPCModel(
            uid, JobQ, ResultQ,
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


class Worker_IPCData_IPCModel(multiprocessing.Process):

    ''' Single "worker" process that processes tasks delivered via queues.

    Attributes
    ----------
    JobQ : multiprocessing.Queue
    ResultQ : multiprocessing.Queue
    '''

    def __init__(self, uid, JobQ, ResultQ, verbose=0, nReps=1):
        ''' Create single worker process, linked to provided queues.
        '''
        super(type(self), self).__init__()  # Required super constructor call
        self.uid = uid
        self.JobQ = JobQ
        self.ResultQ = ResultQ
        self.verbose = verbose
        self.nReps = nReps

    def run(self):
        ''' Perform calcLocalParamsAndSummarize on jobs in JobQ.
        
        Post Condition
        --------------
        After each inner loop, SuffStatBag is written to ResultQ.
        '''

        # Construct iterator with sentinel value of None (for termination)
        jobIterator = iter(self.JobQ.get, None)

        for jobArgs in jobIterator:
            Dslice, hmodel, LPkwargs = jobArgs

            tstart = time.time()
            for rep in range(self.nReps):
                LPslice = hmodel.calc_local_params(Dslice, **LPkwargs)
                SSslice = hmodel.get_global_suff_stats(
                    Dslice, LPslice, **LPkwargs)
            twork = time.time() - tstart

            self.ResultQ.put((SSslice, twork))
            self.JobQ.task_done()

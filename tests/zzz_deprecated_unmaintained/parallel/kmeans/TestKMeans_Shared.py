"""
Shared memory parallel implementation of k-means local step.

Classes
--------

SharedMemWorker : subclass of Process
    Defines work to be done by a single "worker" process
    which is created with references to shared read-only data
    We assign this process "jobs" via a queue, and read its results
    from a separate results queue.

Test : subclass of unittest.TestCase
    Defines a single problem to solve:
    local step on particular dataset X with parameters Mu
    Provides baseline, serial, and parallel solutions.
    * Baseline: monolithic local step.
    * Serial: perform local step on slices of data in series, then aggregate.
    * Parallel: assign slices to worker processes, aggregate from queue.
"""

import sys
import os
import multiprocessing
from multiprocessing import sharedctypes
import itertools
import numpy as np
import unittest
import ctypes
import time

import bnpy
from KMeansUtil import calcLocalParamsAndSummarize
from KMeansUtil import sliceGenerator
from KMeansUtil import getPtrForArray
from KMeansUtil import runBenchmarkAcrossProblemSizes


class SharedMemWorker(multiprocessing.Process):

    """ Single "worker" process that processes tasks delivered via queues
    """

    def __init__(self, uid, JobQueue, ResultQueue,
                 Xsh=None,
                 Msh=None,
                 returnVal='SuffStatBag',
                 sleepPerUnit=0,
                 verbose=0):
        super(type(self), self).__init__()  # Required super constructor call
        self.uid = uid
        self.JobQueue = JobQueue
        self.ResultQueue = ResultQueue
        self.Xsh = Xsh
        self.Msh = Msh
        self.returnVal = returnVal
        self.sleepPerUnit = sleepPerUnit
        self.verbose = verbose

    def printMsg(self, msg):
        if self.verbose:
            for line in msg.split("\n"):
                print("#%d: %s" % (self.uid, line))

    def run(self):
        # self.printMsg("process SetUp! pid=%d" % (os.getpid()))

        # Construct iterator with sentinel value of None (for termination)
        jobIterator = iter(self.JobQueue.get, None)

        # Loop over tasks in the job queue
        for jobArgs in jobIterator:
            start, stop = jobArgs
            SS = calcLocalParamsAndSummarize(self.Xsh, self.Msh,
                                      start=start, stop=stop, 
                                      returnVal=self.returnVal,
                                      sleepPerUnit=self.sleepPerUnit)
            self.ResultQueue.put(SS)
            self.JobQueue.task_done()

        # Clean up
        # self.printMsg("process CleanUp! pid=%d" % (os.getpid()))


class Test(unittest.TestCase):

    def shortDescription(self):
        return None

    def __init__(self, testname, 
                 N=1000, D=25, K=10, nWorkers=2, 
                 sleepPerUnit=0,
                 returnVal='SuffStatBag', verbose=1, **kwargs):
        ''' Create dataset X, cluster means Mu.

        Post Condition Attributes
        --------------
        X : 2D array, N x D
        Mu : 2D array, K x D
        '''
        super(type(self), self).__init__(testname)
        self.nWorkers = nWorkers
        self.verbose = verbose
        self.sleepPerUnit = sleepPerUnit
        self.N = N
        self.D = D
        self.K = K

        self.returnVal = returnVal
        rng = np.random.RandomState((D * K) % 1000)
        self.X = rng.rand(N, D)
        self.Mu = rng.rand(K, D)
        self.Xsh = toSharedMemArray(self.X)
        self.Msh = toSharedMemArray(self.Mu)


    def setUp(self):
        # Create a JobQ (to hold tasks to be done)
        # and a ResultsQ (to hold results of completed tasks)
        manager = multiprocessing.Manager()
        self.JobQ = manager.Queue()
        self.ResultQ = manager.Queue()

        # Launch desired number of worker processes
        # We don't need to store references to these processes,
        # We can get everything we need from JobQ and ResultsQ
        # SHARED MEM: we need to give workers access to shared memory at
        # startup
        for uid in range(self.nWorkers):
            SharedMemWorker(
                uid, self.JobQ, self.ResultQ,
                Xsh=self.Xsh,
                Msh=self.Msh,
                returnVal=self.returnVal,
                sleepPerUnit=self.sleepPerUnit,
                verbose=self.verbose).start()

    def tearDown(self):
        """ Shut down all the workers.
        """
        self.shutdownWorkers()
        time.sleep(0.1)  # let workers all shut down before we quit

    def shutdownWorkers(self):
        """ Shut down all worker processes.
        """
        for workerID in range(self.nWorkers):
            # Passing None to JobQ is shutdown signal
            self.JobQ.put(None)

    def run_baseline(self):
        """ Execute on entire matrix (no slices) in master process.
        """
        SSall = calcLocalParamsAndSummarize(
            self.Xsh, self.Msh, 
            sleepPerUnit=self.sleepPerUnit,
            returnVal=self.returnVal)
        return SSall

    def run_serial(self):
        """ Execute on slices processed in serial by master process.
        """
        N = self.X.shape[0]
        SSagg = None
        for start, stop in sliceGenerator(N, self.nWorkers):
            SSslice = calcLocalParamsAndSummarize(
                self.X, self.Mu, start, stop, 
                sleepPerUnit=self.sleepPerUnit,
                returnVal=self.returnVal)
            if start == 0:
                SSagg = SSslice
            else:
                SSagg += SSslice
        return SSagg

    def run_parallel(self):
        """ Execute on slices processed by workers in parallel.
        """
        # MAP step
        # Create several tasks (one per worker) and add to job queue
        N = self.X.shape[0]
        for start, stop in sliceGenerator(N, self.nWorkers):
            # SHARED MEM means we only put start/stop ids on queue
            # This is much cheaper (hopefully) for inter-proc communication
            self.JobQ.put((start, stop))

        # WAIT 
        # It is crucial to force main thread to sleep now,
        # so other processes can take over the CPU
        self.JobQ.join()

        # REDUCE step
        # Aggregate results across across all workers
        nDone = 0
        SS = 0
        while (nDone < self.nWorkers):
            if not self.ResultQ.empty():
                SSchunk = self.ResultQ.get()
                if nDone == 0:
                    SS = SSchunk
                else:
                    SS += SSchunk
                nDone += 1
            else:
                time.sleep(0.02) # wait 2 ms before checking again
        return SS

    def test_correctness_serial(self):
        ''' Verify that the local step works as expected.

        No parallelization here.
        Just verifying that we can split computation up into >1 slice,
        add up results from all slices and still get the same answer.
        '''
        print('')
        SS1 = self.run_baseline()
        SS2 = self.run_serial()
        assert_SS_allclose(SS1, SS2)

    def test_correctness_parallel(self):
        """ Verify that we can execute local step across several processes

        Each process does the following:
        * grab its chunk of data from a shared jobQueue
        * performs computations on this chunk
        * load the resulting suff statistics object into resultsQueue
        """
        print('')
        SS1 = self.run_parallel()
        SS2 = self.run_baseline()
        assert_SS_allclose(SS1, SS2)

    def run_speed_benchmark(self, method='all', nRepeat=3):
        """ Compare speed of different algorithms.
        """
        if method == 'all':
            Results = self.run_all_with_timer(nRepeat=nRepeat)
        elif method == 'parallel':
            ptime = self.run_with_timer('run_parallel', nRepeat=nRepeat)
            Results = dict(parallel_time=ptime)

        for key in ['base_time', 'serial_time', 'parallel_time']:
            if key in Results:
                try:
                    speedupval = Results[key.replace('time', 'speedup')]
                    speedupmsg = "| %8.3f speedup" % (speedupval)
                except KeyError:
                    speedupmsg = ""
                print("%18s | %8.3f sec %s" % (
                    key,
                    Results[key],
                    speedupmsg
                ))
        return Results

    def run_with_timer(self, funcToCall, nRepeat=3):
        """ Timing experiment specified by funcToCall.
        """
        starttime = time.time()
        for r in range(nRepeat):
            getattr(self, funcToCall)()
        return (time.time() - starttime) / nRepeat

    def run_all_with_timer(self, nRepeat=3):
        """ Timing experiments with baseline, serial, and parallel versions.
        """
        print('BASE----------------')
        base_time = self.run_with_timer('run_baseline', nRepeat)
        time.sleep(0.3)
        print('\nSERIAL--------------')
        serial_time = self.run_with_timer('run_serial', nRepeat)
        time.sleep(0.3)
        
        print('\nPARALLEL------------')
        parallel_time = self.run_with_timer('run_parallel', nRepeat)

        return dict(
            base_time=base_time,
            base_speedup=1.0,
            serial_time=serial_time,
            serial_speedup=base_time / serial_time,
            parallel_time=parallel_time,
            parallel_speedup=base_time / parallel_time,
        )


def toSharedMemArray(X):
    """ Get copy of X accessible from shared memory

    Returns
    --------
    Xsh : RawArray (same size as X)
        Uses separate storage than original array X.
    """
    Xtmp = np.ctypeslib.as_ctypes(X)
    Xsh = multiprocessing.sharedctypes.RawArray(Xtmp._type_, Xtmp)
    return Xsh


def assert_SS_allclose(SS1, SS2):
    if hasattr(SS1, 'CountVec'):
        print("  SS1.CountVec = ", SS1.CountVec[:3])
        print("  SS2.CountVec = ", SS2.CountVec[:3])
        assert np.allclose(SS1.CountVec, SS2.CountVec)
        assert np.allclose(SS1.DataStatVec, SS2.DataStatVec)
    else:
        assert SS1 == SS2

if __name__ == "__main__":
    runBenchmarkAcrossProblemSizes(Test)

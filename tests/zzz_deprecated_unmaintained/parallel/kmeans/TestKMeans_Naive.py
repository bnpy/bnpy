"""
Naive first try at parallel implementation of k-means local step.

Classes
--------
Worker : subclass of Process
    Defines work to be done by a single "worker" process.
    We assign this process "jobs" via a queue, and read its results
    from a separate results queue.
    For each job, the worker reads data via inter-proc communication,
    and performs local step on a slice of the data.

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


class Worker(multiprocessing.Process):

    """ Single "worker" process that processes tasks delivered via queue
    """

    def __init__(self, uid, JobQueue, ResultQueue, verbose=0):
        super(type(self), self).__init__()  # Required super constructor call
        self.uid = uid
        self.JobQueue = JobQueue
        self.ResultQueue = ResultQueue
        self.verbose = verbose

    def printMsg(self, msg):
        if self.verbose:
            for line in msg.split("\n"):
                print("#%d: %s" % (self.uid, line))

    def run(self):
        #self.printMsg("process SetUp! pid=%d" % (os.getpid()))

        # Construct iterator with sentinel value of None (for termination)
        jobIterator = iter(self.JobQueue.get, None)

        for jobArgs in jobIterator:
            X, Mu, start, stop = jobArgs
            # if start is not None:
            #    self.printMsg("start=%d, stop=%d" % (start, stop))
            #msg = "X memory location: %d" % (getPtrForArray(X))
            #self.printMsg(msg)

            SS = calcLocalParamsAndSummarize(X, Mu, start=start, stop=stop)
            self.ResultQueue.put(SS)
            self.JobQueue.task_done()

        # Clean up
        # self.printMsg("process CleanUp! pid=%d" % (os.getpid()))


class Test(unittest.TestCase):

    def shortDescription(self):
        return None


    def __init__(self, testname, 
                 N=1000, D=25, K=10, nWorkers=2, 
                 verbose=1, **kwargs):
        ''' Create dataset X, cluster means Mu.

        Post Condition Attributes
        --------------
        X : 2D array, N x D
        Mu : 2D array, K x D
        '''
        super(type(self), self).__init__(testname)
        self.nWorkers = nWorkers
        self.verbose = verbose
        self.N = N
        self.D = D
        self.K = K

        rng = np.random.RandomState((D * K) % 1000)
        self.X = rng.rand(N, D)
        self.Mu = rng.rand(K, D)

    def setUp(self):
        # Create a JobQ (to hold tasks to be done)
        # and a ResultsQ (to hold results of completed tasks)
        manager = multiprocessing.Manager()
        self.JobQ = manager.Queue()
        self.ResultQ = manager.Queue()

        # Launch desired number of worker processes
        # We don't need to store references to these processes,
        # We can get everything we need from JobQ and ResultsQ
        for uid in range(self.nWorkers):
            Worker(uid, self.JobQ, self.ResultQ, verbose=self.verbose).start()

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
        SSall = calcLocalParamsAndSummarize(self.X, self.Mu)
        return SSall

    def run_serial(self):
        """ Execute on slices processed in serial by master process.
        """
        N = self.X.shape[0]
        SSagg = None
        for start, stop in sliceGenerator(N, self.nWorkers):
            SSslice = calcLocalParamsAndSummarize(self.X, self.Mu, start, stop)
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
            self.JobQ.put((self.X[start:stop], self.Mu, None, None))
            # self.JobQ.put((self.X, self.Mu, start, stop))

        # Pause at this line until all jobs are marked complete.
        self.JobQ.join()

        # REDUCE step
        # Aggregate results across across all workers
        SS = self.ResultQ.get()
        while not self.ResultQ.empty():
            SSchunk = self.ResultQ.get()
            SS += SSchunk
        return SS

    def test_correctness_serial(self):
        ''' Verify that the local step works as expected.

        No parallelization here.
        Just verifying that we can split computation up into >1 slice,
        add up results from all slices and still get the same answer.
        '''
        print('')

        # Version A: summarize entire dataset
        SSall = calcLocalParamsAndSummarize(self.X, self.Mu)

        # Version B: summarize each slice separately, then aggregate
        N = self.X.shape[0]
        SSagg = None
        for start, stop in sliceGenerator(N, self.nWorkers):
            SSslice = calcLocalParamsAndSummarize(self.X, self.Mu, start, stop)
            if start == 0:
                SSagg = SSslice
            else:
                SSagg += SSslice

        # Both A and B better give the same answer
        assert np.allclose(SSall.CountVec, SSagg.CountVec)
        assert np.allclose(SSall.DataStatVec, SSagg.DataStatVec)

    def test_correctness_parallel(self):
        """ Verify that we can execute local step across several processes

        Each process does the following:
        * grab its chunk of data from a shared jobQueue
        * performs computations on this chunk
        * load the resulting suff statistics object into resultsQueue
        """
        print('')
        SS = self.run_parallel()

        # Baseline: compute desired answer in master process.
        SSall = calcLocalParamsAndSummarize(self.X, self.Mu)

        print("Parallel Answer: CountVec = ", SS.CountVec[:3])
        print("   Naive Answer: CountVec = ", SSall.CountVec[:3])
        assert np.allclose(SSall.CountVec, SS.CountVec)
        assert np.allclose(SSall.DataStatVec, SS.DataStatVec)

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
        serial_time = self.run_with_timer('run_serial', nRepeat)
        parallel_time = self.run_with_timer('run_parallel', nRepeat)
        base_time = self.run_with_timer('run_baseline', nRepeat)

        return dict(
            base_time=base_time,
            base_speedup=1.0,
            serial_time=serial_time,
            serial_speedup=base_time / serial_time,
            parallel_time=parallel_time,
            parallel_speedup=base_time / parallel_time,
        )


if __name__ == "__main__":
    runBenchmarkAcrossProblemSizes(Test)

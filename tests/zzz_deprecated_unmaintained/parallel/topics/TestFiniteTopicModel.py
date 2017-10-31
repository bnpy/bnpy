"""
Collection of functions and classes for testing a naive version
of parallel execution for topic model variational inference.

Usage
-----
From a shell/terminal, use like a standard Python script.
$ python TestFiniteTopicModel.py --N 200 --nDoc 500 --K 50 --nWorkers 2

For some keyword args (like N, nDoc, K, nWorkers),
can use range syntax to easily compare performance as params change.
$ python TestFiniteTopicModel.py --nWorkers 2-4 --nDoc 1000
will repeat the test with 1000 documents for 2, 3, and 4 workers

Keyword Args
------------
* N : int or range
    number of words per document for toy data
* nDoc : int or range
    total number of documents to generate
* K : int or range
    number of topics
* nWorkers : int or range
    number of worker processes for parallel execution
* method : {'all', 'baseline', 'parallel', 'serial'}. Default = 'all'.
    identifies which style of computation to perform
* nRepeat : int
    number of times to repeat each called method

Methods
-----------
* runBenchmarkAcrossProblemSizes
    Executable function that parses cmd-line args and runs benchmark tests.

* calcLocalParamsAndSummarize
    Execute local and summary step on slice of data.
    This is the function that we wish to call from each parallel worker.

"""
import os
import multiprocessing
from multiprocessing import sharedctypes
import warnings
import numpy as np
import unittest
import ctypes
import time
import itertools

import bnpy


def runBenchmarkAcrossProblemSizes(TestClass):
    """ Execute speed benchmark across several problem sizes.

    This is main function executed by running this file as script.

    Parameters
    --------
    TestClass : constructor for a TestCase instance
        Must offer a run_speed_benchmark method.

    Post Condition
    --------
    Speed tests are executed, and results are printed to std out.
    """
    import argparse

    if 'OMP_NUM_THREADS' not in os.environ:
        os.environ['OMP_NUM_THREADS'] = '1'

    parser = argparse.ArgumentParser()
    parser.add_argument('--nDoc', type=str, default='100')
    parser.add_argument('--N', type=str, default='200')
    parser.add_argument('--K', type=str, default='50')
    parser.add_argument('--nWorkers', type=str, default='2')
    parser.add_argument('--vocab_size', type=int, default=1000)
    parser.add_argument('--nCoordAscentItersLP', type=int, default=100)
    parser.add_argument('--convThrLP', type=float, default=0.001)
    parser.add_argument('--method', type=str, default='all')
    parser.add_argument('--nRepeat', type=int, default=1)
    parser.add_argument('--verbose', type=int, default=0)
    args = parser.parse_args()

    NKDiterator = itertools.product(
        rangeFromString(args.N),
        rangeFromString(args.K),
        rangeFromString(args.nDoc),
        rangeFromString(args.nWorkers))

    kwargs = dict(**args.__dict__)

    print("Speed Test.")
    print("    OMP_NUM_THREADS=%s" % (os.environ['OMP_NUM_THREADS']))
    print("    vocab_size=%d" % (args.vocab_size))
    print("    nCoordAscentItersLP=%d" % (args.nCoordAscentItersLP))
    print("    convThrLP=%.2e" % (args.convThrLP))

    for (N, K, nDoc, nWorkers) in NKDiterator:
        print('========================= nDoc %d  N %d  K=%d | nWorkers %d' \
            % (nDoc, N, K, nWorkers))
        kwargs['N'] = N
        kwargs['K'] = K
        kwargs['nDoc'] = nDoc
        kwargs['nWorkers'] = nWorkers

        # Create test instance with desired keyword args.
        # Required first arg is string name of test we'll execute
        myTest = TestClass('run_speed_benchmark', **kwargs)
        myTest.setUp()
        TimeInfo = myTest.run_speed_benchmark(method=args.method,
                                              nRepeat=args.nRepeat)
        myTest.tearDown()  # closes all processes


def calcLocalParamsAndSummarize(Data, hmodel, start=None, stop=None, **kwargs):
    ''' Execute local step and summary step on slice of data.

    Args
    ----
    Data : bnpy Data object
    hmodel : bnpy HModel
    start : int or None
        id of data item that starts desired slice
    stop : int or None
        id of data item that ends desired slice

    Returns
    -----------
    SS : bnpy SuffStatBag
    '''
    sliceArgs = dict(cslice=(start, stop))
    kwargs.update(sliceArgs)
    LP = hmodel.obsModel.calc_local_params(Data, dict(), **kwargs)
    LP = hmodel.allocModel.calc_local_params(Data, LP, **kwargs)

    SS = hmodel.allocModel.get_global_suff_stats(Data, LP, **sliceArgs)
    SS = hmodel.obsModel.get_global_suff_stats(Data, SS, LP, **sliceArgs)
    return SS


def sliceGenerator(nDoc=0, nWorkers=0):
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
        yield start, stop


class Worker(multiprocessing.Process):

    """ Single "worker" process that processes tasks delivered via queues
    """

    def __init__(self, uid, JobQueue, ResultQueue,
                 Data=None,
                 hmodel=None,
                 LPkwargs=None,
                 verbose=0):
        super(Worker, self).__init__()
        self.uid = uid
        self.Data = Data
        self.hmodel = hmodel
        if LPkwargs is None:
            LPkwargs = dict()
        self.LPkwargs = LPkwargs

        self.JobQueue = JobQueue
        self.ResultQueue = ResultQueue
        self.verbose = verbose

    def printMsg(self, msg):
        if self.verbose:
            for line in msg.split("\n"):
                print("#%d: %s" % (self.uid, line))

    def run(self):
        self.printMsg("process SetUp! pid=%d" % (os.getpid()))

        # Construct iterator with sentinel value of None (for termination)
        jobIterator = iter(self.JobQueue.get, None)

        for jobArgs in jobIterator:
            start, stop = jobArgs

            SS = calcLocalParamsAndSummarize(
                self.Data, self.hmodel,
                start=start, stop=stop, **self.LPkwargs)
            self.ResultQueue.put(SS)
            self.JobQueue.task_done()

        # Clean up
        self.printMsg("process CleanUp! pid=%d" % (os.getpid()))


class Test(unittest.TestCase):

    def shortDescription(self):
        return None

    def __init__(self, testname, seed=0, vocab_size=100,
                 nCoordAscentItersLP=100, convThrLP=0.01,
                 N=1000, nDoc=25, K=10, nWorkers=1, verbose=1,
                 **kwargs):
        ''' Create a new test harness for parallel topic model inference.

        Post Condition Attributes
        --------------
        Data : bnpy DataObj dataset
        hmodel : bnpy HModel
        '''
        super(type(self), self).__init__(testname)
        self.nWorkers = nWorkers
        self.verbose = verbose
        self.N = N
        self.K = K
        self.nDoc = nDoc
        self.LPkwargs = dict(nCoordAscentItersLP=nCoordAscentItersLP,
                             convThrLP=convThrLP)

        PRNG = np.random.RandomState(seed)
        topics = PRNG.gamma(1.0, 1.0, size=(K, vocab_size))
        np.maximum(topics, 1e-30, out=topics)
        topics /= topics.sum(axis=1)[:, np.newaxis]
        topic_prior = 1.0 / K * np.ones(K)
        self.Data = bnpy.data.BagOfWordsData.CreateToyDataFromLDAModel(
            nWordsPerDoc=N, nDocTotal=nDoc, K=K, topics=topics,
            seed=seed, topic_prior=topic_prior)

        self.hmodel = bnpy.HModel.CreateEntireModel(
            'VB', 'FiniteTopicModel', 'Mult',
            dict(alpha=0.1, gamma=5),
            dict(lam=0.1),
            self.Data)
        self.hmodel.init_global_params(self.Data, initname='randexamples', K=K)

    def setUp(self, **kwargs):
        ''' Launch pool of worker processes, with queues to communicate with.
        '''
        # Create a JobQ (to hold tasks to be done)
        # and a ResultsQ (to hold results of completed tasks)
        manager = multiprocessing.Manager()
        self.JobQ = manager.Queue()
        self.ResultQ = manager.Queue()

        # Launch desired number of worker processes
        # We don't need to store references to these processes,
        # We can get everything we need from JobQ and ResultsQ
        for uid in range(self.nWorkers):
            Worker(uid, self.JobQ, self.ResultQ,
                   Data=self.Data,
                   hmodel=self.hmodel,
                   LPkwargs=self.LPkwargs,
                   verbose=self.verbose).start()

    def tearDown(self):
        """ Shut down all the workers.
        """
        self.shutdownWorkers()

    def shutdownWorkers(self):
        """ Shut down all worker processes.
        """
        for workerID in range(self.nWorkers):
            # Passing None to JobQ is shutdown signal
            self.JobQ.put(None)

    def run_baseline(self):
        """ Execute on entire matrix (no slices) in master process.
        """
        SSall = calcLocalParamsAndSummarize(self.Data, self.hmodel,
                                            **self.LPkwargs)
        return SSall

    def run_serial(self):
        """ Execute on slices processed in serial by master process.
        """
        SSagg = None
        for start, stop in sliceGenerator(self.nDoc, self.nWorkers):
            SSslice = calcLocalParamsAndSummarize(
                self.Data, self.hmodel, start, stop,
                **self.LPkwargs)
            if start == 0:
                SSagg = SSslice
            else:
                SSagg += SSslice
        return SSagg

    def run_parallel(self):
        """ Execute on slices processed by workers in parallel.
        """
        # MAP!
        # Create several tasks (one per worker) and add to job queue
        for start, stop in sliceGenerator(self.nDoc, self.nWorkers):
            self.JobQ.put((start, stop))

        # REDUCE!
        # Aggregate results across across all workers
        # Avoids JobQueue.join() call (which blocks execution)
        # Instead lets main process aggregate all results as they come in.
        nTaskDone = 0
        while nTaskDone < self.nWorkers:
            if not self.ResultQ.empty():
                SSchunk = self.ResultQ.get()
                if nTaskDone == 0:
                    SS = SSchunk
                else:
                    SS += SSchunk
                nTaskDone += 1
        # At this point all jobs are marked complete.
        return SS

    def test_correctness_serial(self):
        ''' Verify that the local step works as expected.

        No parallelization here.
        Just verifying that we can split computation up into >1 slice,
        add up results from all slices and still get the same answer.
        '''
        print('')
        SSbase = self.run_baseline()
        SSserial = self.run_serial()
        allcloseSS(SSbase, SSserial)

    def test_correctness_parallel(self):
        """ Verify that we can execute local step across several processes

        Each process does the following:
        * grab its chunk of data from a shared jobQueue
        * performs computations on this chunk
        * load the resulting suff statistics object into resultsQueue
        """
        print('')
        SSparallel = self.run_parallel()
        SSbase = self.run_baseline()
        allcloseSS(SSparallel, SSbase)

    def test_speed(self, nRepeat=5):
        """ Compare speed of different algorithms.
        """
        print('')
        Results = self.run_all_with_timer(nRepeat=nRepeat)
        assert True

    def run_speed_benchmark(self, method='all', nRepeat=3):
        """ Compare speed of different algorithms.
        """
        if method == 'all':
            Results = self.run_all_with_timer(nRepeat=nRepeat)
        elif method == 'parallel':
            ptime = self.run_with_timer('run_parallel', nRepeat=nRepeat)
            Results = dict(parallel_time=ptime)
        elif method == 'serial':
            ptime = self.run_with_timer('run_serial', nRepeat=nRepeat)
            Results = dict(serial_time=ptime)
        else:
            ptime = self.run_with_timer('run_baseline', nRepeat=nRepeat)
            Results = dict(base_time=ptime)

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


def rangeFromString(commaString):
    """ Convert a comma string like "1,5-7" into a list [1,5,6,7]

    Returns
    --------
    myList : list of integers

    Reference
    -------
    http://stackoverflow.com/questions/6405208/\
    how-to-convert-numeric-string-ranges-to-a-list-in-python
    """
    listOfLists = [rangeFromHyphen(r) for r in commaString.split(',')]
    flatList = itertools.chain(*listOfLists)
    return flatList


def rangeFromHyphen(hyphenString):
    """ Convert a hyphen string like "5-7" into a list [5,6,7]

    Returns
    --------
    myList : list of integers
    """
    x = [int(x) for x in hyphenString.split('-')]
    return list(range(x[0], x[-1] + 1))


def allcloseSS(SS1, SS2):
    """ Verify that two suff stat bags have indistinguishable data.
    """
    # Both A and B better give the same answer
    for key in list(SS1._FieldDims.keys()):
        arr1 = getattr(SS1, key)
        arr2 = getattr(SS2, key)
        print(key)
        if isinstance(arr1, float):
            print(arr1)
            print(arr1)
        elif arr1.ndim == 1:
            print(arr1[:3])
            print(arr2[:3])
        else:
            print(arr1[:2, :3])
            print(arr2[:2, :3])
        assert np.allclose(arr1, arr2)


if __name__ == "__main__":
    runBenchmarkAcrossProblemSizes(Test)

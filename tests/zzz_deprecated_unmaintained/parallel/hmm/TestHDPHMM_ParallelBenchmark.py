"""
Collection of functions and classes for testing parallel execution
of the local step for HDPHMM variational inference.

Usage
-----
From a shell/terminal, use like a standard Python script.
$ python TestHDPHMM.py --N 200 --nDoc 500 --K 50 --nWorkers 2

Keyword Args
------------
* nDoc : int or range
    total number of data sequences to generate
* T : int or range
    length of each data sequence
* K : int or range
    number of states in the HMM
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

from bnpy.util import sharedMemDictToNumpy, sharedMemToNumpyArray
from bnpy.learnalg import SharedMemWorker


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
    parser.add_argument('--nDoc', type=str, default='64')
    parser.add_argument('--K', type=str, default='50')
    parser.add_argument('--T', type=int, default='10000')
    parser.add_argument('--dataset', type=str, default='SeqOfBinBars9x9')
    parser.add_argument('--nWorkers', type=str, default='2')
    parser.add_argument('--method', type=str, default='all')
    parser.add_argument('--nRepeat', type=int, default=1)
    parser.add_argument('--verbose', type=int, default=0)
    args = parser.parse_args()

    argIterator = itertools.product(
        rangeFromString(args.nDoc),
        rangeFromString(args.K),
        rangeFromString(args.nWorkers))
    kwargs = dict(**args.__dict__)

    print("Speed Test. %s dataset" % (args.dataset))
    print("    OMP_NUM_THREADS=%s" % (os.environ['OMP_NUM_THREADS']))
    print("    T=%d" % (args.T))
    for (nDoc, K, nWorkers) in argIterator:
        print('========================= nDoc %d  K=%d | nWorkers %d' \
            % (nDoc, K, nWorkers))
        kwargs['nDoc'] = nDoc
        kwargs['K'] = K
        kwargs['nWorkers'] = nWorkers

        # Create test instance with desired keyword args.
        # Required first arg is string name of test we'll execute
        myTest = TestClass('run_speed_benchmark', **kwargs)
        myTest.setUp()
        TimeInfo = myTest.run_speed_benchmark(method=args.method,
                                              nRepeat=args.nRepeat)
        myTest.tearDown()  # closes all processes


class Test(unittest.TestCase):

    def shortDescription(self):
        return None

    def __init__(self, testname, seed=0,
                 dataset='BigChromatinCD4T',
                 T=10000, nDoc=8, K=3, nWorkers=2, verbose=1,
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
        self.T = T
        self.K = K
        self.nDoc = nDoc

        self.Data = loadToyDataset(dataset, nDoc=nDoc, T=T, **kwargs)
        assert self.Data.nDoc == self.nDoc

        # Initialize model
        hmmKwargs = dict(
            alpha=0.5, gamma=10.0, nGlobalSteps=1, nGlobalStepsBigChange=3,
            startAlpha=10, hmmKappa=100)
        self.hmodel = bnpy.HModel.CreateEntireModel(
            'moVB', 'HDPHMM', 'Bern',
            hmmKwargs,
            dict(lam1=0.1, lam0=0.3),
            self.Data)
        self.hmodel.init_global_params(self.Data,
                                       initname='randcontigblocks', K=K)

    def setUp(self, **kwargs):
        ''' Launch pool of worker processes, with queues to communicate with.
        '''
        # Create a JobQ (to hold tasks to be done)
        # and a ResultsQ (to hold results of completed tasks)
        manager = multiprocessing.Manager()
        self.JobQ = manager.Queue()
        self.ResultQ = manager.Queue()

        a_L, a_S = self.hmodel.allocModel.getLocalAndSummaryFunctionHandles()
        o_L, o_S = self.hmodel.obsModel.getLocalAndSummaryFunctionHandles()

        dataSharedMem = self.Data.getRawDataAsSharedMemDict()

        aSharedMem = self.hmodel.allocModel.fillSharedMemDictForLocalStep()
        oSharedMem = self.hmodel.obsModel.fillSharedMemDictForLocalStep()

        # Launch desired number of worker processes
        # We don't need to store references to these processes,
        # We can get everything we need from JobQ and ResultQ
        for uid in range(self.nWorkers):
            worker = SharedMemWorker(uid, self.JobQ, self.ResultQ,
                                     self.Data.getDataSliceFunctionHandle(),
                                     o_L,
                                     o_S,
                                     a_L,
                                     a_S,
                                     dataSharedMem,
                                     aSharedMem,
                                     oSharedMem,
                                     verbose=self.verbose)
            worker.start()

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
        SSall = calcLocalParamsAndSummarize(self.Data, self.hmodel)
        return SSall

    '''
    def run_serial(self):
        """ Execute on slices processed in serial by master process.
        """
        SSagg = None
        aArgs = self.hmodel.allocModel.getSerializableParamsForLocalStep()
        oArgs = self.hmodel.obsModel.getSerializableParamsForLocalStep()
        for jobInfo in sliceGenerator(self.nDoc, self.nWorkers,
                                      aArgs=aArgs, oArgs=oArgs):
            sliceInfo, aArgs, oArgs = jobInfo
            start = sliceInfo['start']
            stop = sliceInfo['stop']

            SSslice = calcLocalParamsAndSummarize(
                self.Data, self.hmodel, start, stop)

            if start == 0:
                SSagg = SSslice
            else:
                SSagg += SSslice
        return SSagg
    '''

    def run_parallel(self):
        """ Execute on slices processed by workers in parallel.
        """
        aArgs = self.hmodel.allocModel.getSerializableParamsForLocalStep()
        oArgs = self.hmodel.obsModel.getSerializableParamsForLocalStep()

        # MAP!
        # Create several tasks (one per worker) and add to job queue
        for jobInfo in sliceGenerator(self.nDoc, self.nWorkers,
                                      aArgs=aArgs, oArgs=oArgs):
            self.JobQ.put(jobInfo)

        self.JobQ.join()
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
        parallel_time = self.run_with_timer('run_parallel', nRepeat)
        base_time = self.run_with_timer('run_baseline', nRepeat)
        serial_time = base_time  # self.run_with_timer('run_serial', nRepeat)

        return dict(
            base_time=base_time,
            base_speedup=1.0,
            serial_time=serial_time,
            serial_speedup=base_time / serial_time,
            parallel_time=parallel_time,
            parallel_speedup=base_time / parallel_time,
        )


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


def sliceGenerator(nDoc=0, nWorkers=0, aArgs=dict(), oArgs=dict()):
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
        yield dict(start=start, stop=stop, batchID=0), aArgs, oArgs


def loadToyDataset(dataset, nDoc=3, T=1000, **kwargs):
    if os.path.exists('/tmp/'):
        tmproot = '/tmp/'
    else:
        tmproot = '/ltmp/'

    if dataset == 'BigChromatinCD4T':
        cachefilename = '%s/%s_nDoc=128.mat' % (tmproot, dataset)
    else:
        cachefilename = '%s/%s_nDoc=%d_T=%d.mat' % (tmproot, dataset, nDoc, T)

    if os.path.exists(cachefilename):
        print('Loading dataset from disk...')
        stime = time.time()
        Data = bnpy.data.GroupXData.LoadFromFile(cachefilename)
        print(' done after %.1f sec' % (time.time() - stime))

    elif dataset == 'SeqOfBinBars9x9':
        print('Creating toy dataset with nDoc=%d, T=%d, D=81' % (nDoc, T))
        print('...', end=' ')
        stime = time.time()
        import SeqOfBinBars9x9
        Data = SeqOfBinBars9x9.get_data(nDocTotal=nDoc, T=T)
        print(' done after %.1f sec' % (time.time() - stime))
        Data.save_to_mat(cachefilename)
    elif dataset == 'BigChromatinCD4T':
        import glob
        print('Creating big dataset with nDoc=%d' % (nDoc))
        print('...', end=' ')
        stime = time.time()
        dataPath = '/data/liv/biodatasets/CD4TCellLine/wholegenomebatches/'
        batchpaths = glob.glob(dataPath + 'chr*batch*.mat')
        Data = bnpy.data.GroupXData.LoadFromFile(batchpaths[0])
        for ff, fpath in enumerate(batchpaths[1:128]):
            BatchData = bnpy.data.GroupXData.LoadFromFile(fpath)
            Data.add_data(BatchData)
            print(ff)
        print(' done after %.1f sec' % (time.time() - stime))
        Data.save_to_mat(cachefilename)
    assert Data.nDoc >= nDoc
    if Data.nDoc > nDoc:
        Data = Data.select_subset_by_mask(np.arange(nDoc))
    return Data


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
            # print arr1.sum()
            # print arr2.sum()
        else:
            print(arr1[:2, :3])
            print(arr2[:2, :3])
        assert np.allclose(arr1, arr2)


if __name__ == "__main__":
    runBenchmarkAcrossProblemSizes(Test)

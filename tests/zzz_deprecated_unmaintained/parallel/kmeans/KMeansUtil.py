""" Provides functions useful for different Kmeans implementations

List of functions
--------------

calcLocalParamsAndSummarize
    perform kmeans local step (assignments and summary stats)
    for a "slice" of the provided dataset.

sliceGenerator
    generate disjoint slices for provided dataset, given number of workers

runBenchmarkAcrossProblemSizes
    Time execution of provided implementation, across range of N/D/K vals.
"""
import warnings
import itertools
import numpy as np
import bnpy
import time

def calcLocalParamsAndSummarize(X, Mu, 
                                start=None, stop=None, 
                                returnVal='SuffStatBag',
                                sleepPerUnit=0):
    ''' K-means step

    Returns
    -----------
    SuffStatBag with fields
    * CountVec : 1D array, size K
    * DataStatVec : 2D array, K x D
    '''
    # If needed, convert input arrays from shared memory to numpy format
    if not isinstance(X, np.ndarray):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            Mu = np.ctypeslib.as_array(Mu)
            X = np.ctypeslib.as_array(X)
            # This does *not* allocate any new memory,
            # just allows using X and Mu as numpy arrays
    # Unpack problem size variables
    K, D = Mu.shape

    # Grab current slice (subset) of X to work on
    if start is not None:
        start = int(start)
        stop = int(stop)
        Xcur = X[start:stop]
    else:
        Xcur = X
        start = 0
        stop = X.shape[0]

    if returnVal.count('local'):
        Xcur = Xcur.copy()
    #ptr, read_only_flag = X.__array_interface__['data']
    #print "X    ptr: %s, readonly=%d" % (hex(ptr), read_only_flag)
    #ptr, read_only_flag = Xcur.__array_interface__['data']
    #print "Xcur ptr: %s, readonly=%d" % (hex(ptr), read_only_flag)
    
    tstart = time.time()
    if sleepPerUnit > 0:
        if start is None:
            nUnits = X.shape[0]
        else:
            nUnits = stop - start
        time.sleep(nUnits * sleepPerUnit)
        return 0
    elif sleepPerUnit == -1:
        # CPU-bound stuff
        s = 0
        for i in range(20*(stop - start)):
             s += 1
        telapsed = time.time() - tstart
        print("BIG SUM on slice %d-%d took %.2f sec" % (start, stop, telapsed))
        return 0
    elif sleepPerUnit == -1.5:
        s = 0
        for i in range(50):
            s += Xcur.sum()
        telapsed = time.time() - tstart
        print("SHARED MEM SUM on slice %d-%d took %.2f sec" % (start, stop, telapsed))
        return 0

    elif sleepPerUnit == -1.6:
        s = 0
        for i in range(50):
            s += np.sum(Xcur)
        telapsed = time.time() - tstart
        print("LOCAL COPY SUM on slice %d-%d took %.2f sec" % (start, stop, telapsed))
        return 0
    elif sleepPerUnit == -1.7:
        s = 0
        for i in range(0, 5*(stop-start)):
            s += Xcur[0,0]
        telapsed = time.time() - tstart
        print("FIRST ENTRY SUM on slice %d-%d took %.2f sec" % (start, stop, telapsed))
        return 0 

    elif sleepPerUnit == -1.8:
        s = 0
        N = stop - start
        for i in range(0, 5*N):
            s += Xcur[i % N,0]
        telapsed = time.time() - tstart
        print("COL 1 SUM on slice %d-%d took %.2f sec" % (start, stop, telapsed))
        return 0
    elif sleepPerUnit == -2:
        Dist = np.dot(Xcur, Mu.T)
        telapsed = time.time() - tstart
        print("MAT PROD on slice %d-%d took %.2f sec" % (start, stop, telapsed))
        return 0
    elif sleepPerUnit == -1.9:
        for i in range(75):
            s = Xcur[:, 0].sum()
        telapsed = time.time() - tstart
        print("COL 1 SUM VEC, slice %d-%d took %.2f sec" % (start, stop, telapsed))

    # Dist : 2D array, size N x K
    #     squared euclidean distance from X[n] to Mu[k]
    #     up to an additive constant independent of Mu[k]
    Dist = -2 * np.dot(Xcur, Mu.T)
    Dist += np.sum(np.square(Mu), axis=1)[np.newaxis, :]
    # Z : 1D array, size N
    #     Z[n] gives integer id k of closest cluster cntr Mu[k] to X[n,:]
    Z = Dist.argmin(axis=1)

    CountVec = np.zeros(K)
    DataStatVec = np.zeros((K, D))
    for k in range(K):
        mask_k = Z == k
        CountVec[k] = np.sum(mask_k)
        DataStatVec[k] = np.sum(Xcur[mask_k], axis=0)

    SS = bnpy.suffstats.SuffStatBag(K=K, D=D)
    SS.setField('CountVec', CountVec, dims=('K'))
    SS.setField('DataStatVec', DataStatVec, dims=('K', 'D'))
    
    if returnVal == 'SuffStatBag':
        return SS
    else:
        return SS.CountVec.sum()

def sliceGenerator(N=0, nWorkers=0, nTaskPerWorker=1):
    """ Iterate over slices given problem size and num workers

    Yields
    --------
    (start,stop) : tuple
    """
    taskSize = np.floor(N / nWorkers)
    nTask = nWorkers * nTaskPerWorker
    for taskID in range(nTask):
        start = taskID * taskSize
        stop = (taskID + 1) * taskSize
        if taskID == nTask - 1:
            stop = N
        yield start, stop


def runBenchmarkAcrossProblemSizes(TestClass):
    """ Execute speed benchmark across several N/D/K values.

    Parameters
    --------
    TestClass : constructor for a TestCase instance
        Must offer a run_speed_benchmark method.

    Post Condition
    --------
    Speed tests are executed, and results are printed to std out.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--D', type=str, default='25')
    parser.add_argument('--N', type=str, default='10000')
    parser.add_argument('--K', type=str, default='10')
    parser.add_argument('--nWorkers', type=int, default=2)
    parser.add_argument('--method', type=str, default='parallel')
    parser.add_argument('--nRepeat', type=int, default=1)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--returnVal', type=str, default='SuffStatBag')
    parser.add_argument('--sleepPerUnit', type=float, default=0)
    args = parser.parse_args()

    NKDiterator = itertools.product(
        rangeFromString(args.N),
        rangeFromString(args.K),
        rangeFromString(args.D))

    kwargs = dict(**args.__dict__)

    for (N, K, D) in NKDiterator:
        print('=============================== N=%d K=%d D=%d' % (
            N, K, D))
        kwargs['N'] = N
        kwargs['K'] = K
        kwargs['D'] = D

        # Create test instance with desired keyword args.
        # Required first arg is string name of test we'll execute
        myTest = TestClass('run_speed_benchmark', **kwargs)
        myTest.setUp()
        time.sleep(0.2) # give processes time to launch

        TimeInfo = myTest.run_speed_benchmark(method=args.method,
                                              nRepeat=args.nRepeat)
        myTest.tearDown()  # closes all processes


def getPtrForArray(X):
    """ Get int pointer to memory location of provided array

    This can be used to confirm that different processes are
    accessing a common resource, and not duplicating that resource,
    which is wasteful and slows down execution.

    Returns
    --------
    ptr : int
    """
    if isinstance(X, np.ndarray):
        ptr, read_only_flag = X.__array_interface__['data']
        return int(ptr)
    else:
        return id(X)


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

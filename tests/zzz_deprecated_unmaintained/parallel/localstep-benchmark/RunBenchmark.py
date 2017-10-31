import argparse
import numpy as np
import time
import itertools
import os
import sys
import bnpy

import LocalStepUtil_Baseline

def runBenchmark():
    """ Execute speed benchmark 

    Post Condition
    --------
    Results are printed to std out.

    Returns
    -------
    TimeInfo : info about timing results.
    """
    # Force only single thread to be used for BLAS/Linpack ops like 
    # matrix multiples, etc.
    os.environ['OMP_NUM_THREADS'] = '1'

    # Define expected args and parse from standard input (stdin)
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default='10')
    parser.add_argument('--N', type=int, default='10000',
                        help='number of observations in DPGMM problem')
    parser.add_argument('--D', type=int, default='10',
                        help='dimension of DPGMM problem')
    parser.add_argument('--nDocTotal', type=int, default='1000')
    parser.add_argument('--nWordsPerDoc', type=int, default='200')
    parser.add_argument('--vocab_size', type=int, default='8000')
    parser.add_argument('--nCoordAscentItersLP', type=int, default='100')
    parser.add_argument('--convThrLP', type=float, default='0.000001')
    parser.add_argument('--restartLP', type=int, default='0')
    parser.add_argument('--nWorkers', type=str, default='1')
    parser.add_argument('--showCountVec', type=int, default=0)
    parser.add_argument('--problemSpecModule', type=str,
                        default='ProblemSpec_HDPTopicModel')
    parser.add_argument('--parallelModule', type=str,
                        default='LocalStepUtil_ParallelIPC')
    parser.add_argument('--minDurationPerWorker_sec', type=float, default=1.0)
    parser.add_argument('--nRepsForMinDuration', type=int, default=None)
    parser.add_argument('--verbose', type=int, default=1)
    args = parser.parse_args()

    # Transform input args into a key/value dictionary
    kwargs = dict(**args.__dict__)
    kwargs['LPkwargs'] = dict(
        restartLP=args.restartLP,
        convThrLP=args.convThrLP,
        nCoordAscentItersLP=args.nCoordAscentItersLP,
        doPrecompEntropy=1,
        )
    kwargs['maxWorker'] = np.asarray(rangeFromString(args.nWorkers)).max()


    # Import problem-specific modules
    LocalStepUtil_Parallel = __import__(
        args.parallelModule, fromlist=[])
    ProblemSpec = __import__(
        args.problemSpecModule, fromlist=[])

    # Create Data and model
    Data = ProblemSpec.makeData(**kwargs)
    hmodel = ProblemSpec.makeModel(Data=Data, **kwargs)

    # Determine how many repetitions to do for the task,
    # so that specified minimum duration (if any) is reached.
    kwargs['nRepsForMinDuration'] = calcNumRepsForMinDuration(
        Data=Data, hmodel=hmodel, **kwargs)

    # Pretty-print the problem size information
    printProblemSpec(ProblemSpec=ProblemSpec, **kwargs)

    # Evaluate the baseline method (single-thread)
    tstart = time.time()
    for rep in range(kwargs['nRepsForMinDuration']):
        SS_baseline = LocalStepUtil_Baseline.calcLocalParamsAndSummarize(
            Data=Data, hmodel=hmodel, 
            **kwargs)
    t_baseline = time.time() - tstart
    printTaskResult(t_baseline, 
                    SS=SS_baseline,
                    name='baseline',
                    **kwargs)

    Twork = list()
    Ttotal = list()
    for nWorker in rangeFromString(args.nWorkers):
        kwargs['nWorker'] = nWorker

        # Create pool of worker processes
        # NOTE: pool is made "from scratch" for each value of nWorker
        JobQ, ResultQ = LocalStepUtil_Parallel.setUpWorkers(
            Data=Data, hmodel=hmodel, 
            **kwargs)

        # Perform local/summary step, assigning each process a slice of work
        tstart = time.time()
        SS, twork = LocalStepUtil_Parallel.calcLocalParamsAndSummarize(
            Data=Data, hmodel=hmodel, 
            JobQ=JobQ, ResultQ=ResultQ, 
            **kwargs)
        ttotal = time.time() - tstart

        # Record time required to complete assigned work
        Twork.append(twork)
        Ttotal.append(ttotal)
        printTaskResult(ttotal, twork,
                        SS=SS,
                        name=args.parallelModule,
                        **kwargs)

        # Close pool of worker processes
        LocalStepUtil_Parallel.tearDownWorkers(
            JobQ=JobQ, ResultQ=ResultQ, **kwargs)

def printTaskResult(ttotal, 
        twork=0, nWorker=0, name='baseline',
        SS=None, showCountVec=0, **kwargs):
    if nWorker == 0:
        print('%s | %.2f sec total' % (
            name, ttotal))
    else:
        msg = '%s: %d workers | %.2f sec total'
        msg += ' | %.2f sec max single worker no overhead'
        msg = msg % (
            name.replace('LocalStepUtil_', ''), nWorker, ttotal, twork)
        print(msg)
    if SS is not None and showCountVec:
        countVecStr = ' '.join(['%.1f' % (Nk) for Nk in SS.getCountVec()[:10]])
        print("  countvec ", countVecStr)


def printProblemSpec(
        ProblemSpec=None,
        minDurationPerWorker_sec=0, nRepsForMinDuration=0, **kwargs):
    ''' Print out info about how we scale computation to reach min duration.
    '''
    print('Task Specification')
    print('------------------')
    print('OMP_NUM_THREADS=%s' % (os.environ['OMP_NUM_THREADS']))
    print(ProblemSpec.pprintProblemSpecStr(**kwargs))
    print('Minimum duration per worker: %.1f sec' % (minDurationPerWorker_sec))
    print('To achieve min duration, we will repeat each task %d times' % (
        nRepsForMinDuration))
    print('')

def calcNumRepsForMinDuration(Data=None, hmodel=None, **kwargs):
    ''' Compute number of times to repeat local step to reach minimum duration.

    Effective parallelization requires that overhead from IPC/startup/etc
    is a small fraction of total time spent (thus minimum duration)

    Returns
    -------
    nReps : int
    '''
    nReps = kwargs['nRepsForMinDuration']
    if nReps is None or nReps == 'None':
        minDurationPerWorker_sec = kwargs['minDurationPerWorker_sec']
        maxWorker = kwargs['maxWorker']
        minDuration_sec = maxWorker * minDurationPerWorker_sec
        if minDurationPerWorker_sec > 0:
            tstart = time.time()
            SS_baseline = LocalStepUtil_Baseline.calcLocalParamsAndSummarize(
                Data=Data, hmodel=hmodel, 
                **kwargs)
            t_baseline = time.time() - tstart
            if t_baseline < minDuration_sec:
                nReps = int(np.ceil(
                    minDuration_sec / t_baseline))
            else:
                nReps = 1
        else:
            nReps = 1
    return nReps

def sliceGenerator(Data=None, nWorker=0, nTaskPerWorker=1, **kwargs):
    ''' Iterate over slices given problem size and num workers

    Yields
    --------
    (start,stop) : tuple
    '''
    sliceSize = np.floor(Data.get_size() / nWorker)
    for sliceID in range(nWorker * nTaskPerWorker):
        start = sliceID * sliceSize
        stop = (sliceID + 1) * sliceSize
        if sliceID == (nWorker * nTaskPerWorker) - 1:
            stop = Data.get_size()
        yield start, stop


def rangeFromString(commaString):
    ''' Convert a comma string like "1,5-7" into a list [1,5,6,7]

    Returns
    --------
    myList : list of integers

    Reference
    -------
    http://stackoverflow.com/questions/6405208/\
    how-to-convert-numeric-string-ranges-to-a-list-in-python
    '''
    listOfLists = [rangeFromHyphen(r) for r in commaString.split(',')]
    flatList = itertools.chain(*listOfLists)
    return [x for x in flatList]


def rangeFromHyphen(hyphenString):
    ''' Convert a hyphen string like "5-7" into a list [5,6,7]

    Returns
    --------
    myList : list of integers
    '''
    x = [int(x) for x in hyphenString.split('-')]
    return list(range(x[0], x[-1] + 1))

if __name__ == '__main__':
    runBenchmark()

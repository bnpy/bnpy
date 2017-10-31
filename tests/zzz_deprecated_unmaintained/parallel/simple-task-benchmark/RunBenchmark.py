import numpy as np
import multiprocessing
import argparse
import os
import itertools
import time
from collections import defaultdict
from matplotlib import pylab
import copy
import sys
import platform

ColorMap = dict(monolithic='k', serial='b', parallel='r')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=str, default='10000')
    parser.add_argument('--D', type=str, default='25')
    parser.add_argument('--nWorker', type=str, default='1')
    parser.add_argument('--methods', type=str, default='monolithic,parallel')
    parser.add_argument('--nRepeat', type=int, default=1)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--task', type=str, default='sleep')
    parser.add_argument('--minSliceDuration', type=float, default=1.0)
    parser.add_argument('--scaleFactor', type=float, default=0.0)
    parser.add_argument('--memoryType', type=str, default='shared')
    parser.add_argument('--savefig', type=int, default=0)
    parser.add_argument(
        '--maxWorker',
        type=int,
        default=multiprocessing.cpu_count())
    args = parser.parse_args()

    nWorker_IN = args.nWorker + ''
    pargs = args.__dict__
    pargs['methods'] = args.methods.split(',')

    for task in args.task.split(','):
        pargs['task'] = task
        AllInfo = initBenchmarkInfo(**pargs)
        pargs['scaleFactor'] = 0.0

        for probSizeArgs in problemGenerator(**pargs):
            kwargs = copy.deepcopy(pargs)
            kwargs.update(probSizeArgs)
            X = makeData(**kwargs)

            if np.allclose(pargs['scaleFactor'], 0) and args.task != 'sleep':
                pargs['scaleFactor'] = getScaleFactorForTask(X, **kwargs)
                kwargs['scaleFactor'] = pargs['scaleFactor']

            JobQ, ResultQ = launchWorkers(X, **kwargs)
            time.sleep(.1)

            CurInfo = runAllBenchmarks(X, JobQ, ResultQ, **kwargs)
            AllInfo = updateBenchmarkInfo(AllInfo, CurInfo, **kwargs)

            closeWorkers(JobQ, **kwargs)
            time.sleep(.1)

        pylab.figure(1)
        pylab.xlim([-0.2, kwargs['maxWorker'] + 0.5])
        pylab.ylim(
            [-0.002, kwargs['minSliceDuration'] * kwargs['maxWorker'] * 2.2])
        pylab.legend(loc='upper left')
        pylab.xlabel('Number of Workers')
        pylab.ylabel('Wallclock time')

        hostname = platform.node()
        plotSpeedupFigure(AllInfo, **kwargs)
        if args.savefig:
            pylab.figure(1)
            title = 'BenchmarkPlot_%s_%s_minDur=%.2f_WallclockTimes.eps'\
                % (hostname, task, kwargs['minSliceDuration'])
            pylab.savefig(title,
                          format='eps',
                          bbox_inches='tight',
                          pad_inches=0)

            pylab.figure(2)
            title = 'BenchmarkPlot_%s_%s_minDur=%.2f_Speedup.eps'\
                % (hostname, task, kwargs['minSliceDuration'])
            pylab.savefig(title,
                          format='eps',
                          bbox_inches='tight',
                          pad_inches=0)

        if not args.savefig:
            pylab.show(block=1)
        pylab.close('all')


def getMethodNames(methods='all', **kwargs):
    allMethodNames = ['monolithic', 'serial', 'parallel']
    methodNames = list()
    for name in allMethodNames:
        if 'all' not in methods and name not in methods:
            continue
        methodNames.append(name)
    return methodNames


def plotSpeedupFigure(AllInfo, maxWorker=1, **kwargs):
    pylab.figure(2)
    xs = AllInfo['nWorker']
    ts_mono = AllInfo['t_monolithic']

    xgrid = np.linspace(0, maxWorker + 0.1, 100)
    pylab.plot(xgrid, xgrid, 'y--', label='ideal parallel')

    for method in getMethodNames(**kwargs):
        speedupRatio = ts_mono / AllInfo['t_' + method]
        pylab.plot(xs, speedupRatio, 'o-',
                   label=method,
                   color=ColorMap[method],
                   markeredgecolor=ColorMap[method])

    pylab.xlim([-0.2, maxWorker + 0.5])
    pylab.ylim([0, maxWorker + 0.5])
    pylab.legend(loc='upper left')
    pylab.xlabel('Number of Workers')
    pylab.ylabel('Speedup over Monolithic')


def getScaleFactorForTask(X, maxWorker=1, **kwargs):

    minSliceDuration = kwargs['minSliceDuration']
    N = X.shape[0]
    sliceSize = np.floor(N / maxWorker)
    Xslice = X[:sliceSize]
    kwargs['scaleFactor'] = 1
    print('FINDING PROBLEM SCALE.')
    print('  Max possible workers %d\n Min duration of slice: %.2f' \
        % (maxWorker, minSliceDuration))

    t = workOnSlice(Xslice, None, None, **kwargs)

    while t < minSliceDuration:
        kwargs['scaleFactor'] *= 2
        t = workOnSlice(Xslice, None, None, **kwargs)
    print('SCALE: ', kwargs['scaleFactor'], 'telapsed=%.3f' % (t))
    return kwargs['scaleFactor']


def initBenchmarkInfo(**kwargs):
    AllInfo = dict()
    AllInfo['nWorker'] = np.asarray(
        [float(x) for x in rangeFromString(kwargs['nWorker'])])
    for name in getMethodNames(**kwargs):
        AllInfo['t_' + name] = np.zeros_like(AllInfo['nWorker'])
    return AllInfo


def updateBenchmarkInfo(AllInfo, CurInfo, nRepeat, **kwargs):
    """ Aggregate information about different experiments into one dict.
    """
    for method in CurInfo:
        tvec = np.asarray([
            CurInfo[method][r]['telapsed'] for r in range(nRepeat)])
        key = 't_' + method
        loc = np.flatnonzero(AllInfo['nWorker'] == kwargs['nWorker'])
        AllInfo[key][loc] = np.median(tvec)
    return AllInfo


def workOnSlice(X, start, stop,
                task='sleep',
                minSliceDuration=1.0,
                memoryType='shared',
                nWorker=1,
                maxWorker=1,
                scaleFactor=1.0,
                **kwargs):
    """ Perform work on a slice of data.
    """
    if start is None:
        start = 0
        stop = X.shape[0]
        Xslice = X
    else:
        start = int(start)
        stop = int(stop)
        Xslice = X[start:stop]

    if memoryType == 'local':
        Xslice = Xslice.copy()
    elif memoryType == 'random':
        Xslice = np.random.rand(Xslice.shape)
    nReps = int(np.ceil(scaleFactor))

    tstart = time.time()
    if task == 'sleep':
        # Make sure that we sleep same total amt whether serial or parallel
        duration = minSliceDuration * (maxWorker / float(nWorker))
        time.sleep(duration)
    elif task == 'sumforloop':
        for rep in range(nReps):
            s = 0
            for n in range(stop - start):
                s = 2 * n
    elif task == 'Xcolsumforloop':
        for rep in range(nReps):
            s = 0.0
            for n in range(stop - start):
                s += Xslice[n, 0]
    elif task == 'Xcolsumvector':
        for rep in range(nReps):
            s = Xslice[:, 0].sum()

    elif task == 'Xelementwiseproduct':
        for rep in range(nReps):
            S = Xslice * Xslice
    elif task == 'Xmatrixproduct':
        for rep in range(nReps):
            S = np.dot(Xslice.T, Xslice)
    else:
        raise NotImplementedError("Unrecognized task: %s" % (task))
    telapsed = time.time() - tstart
    return telapsed


def runBenchmark(X, JobQ, ResultQ,
                 method='serial',
                 nWorker=1, nTaskPerWorker=1,
                 **kwargs):
    kwargs = dict(**kwargs)
    N = X.shape[0]
    ts = list()
    if method == 'monolithic':
        t = workOnSlice(X, None, None, **kwargs)
        ts.append(t)
    elif method == 'serial':
        kwargs['nWorker'] = nWorker  # scale work load by num workers
        for start, stop in sliceGenerator(N, nWorker, nTaskPerWorker):
            t = workOnSlice(X, start, stop, **kwargs)
            ts.append(t)
    elif method == 'parallel':
        kwargs['nWorker'] = nWorker  # scale work load by num workers
        for start, stop in sliceGenerator(N, nWorker, nTaskPerWorker):
            JobQ.put((start, stop, kwargs))

        JobQ.join()
        while not ResultQ.empty():
            t = ResultQ.get()
            ts.append(t)
    return ts


def runAllBenchmarks(X, JobQ, ResultQ,
                     nRepeat=1, methods='all', **kwargs):
    methodNames = getMethodNames(methods=methods)

    print('======================= ', makeTitle(**kwargs))
    print('%16s %15s %15s %15s %10s' % (
        ' ',  'slice size', 'slice time', 'wallclock time', 'speedup'))
    Tinfo = defaultdict(dict)
    for rep in range(nRepeat):
        for method in methodNames:
            tstart = time.time()
            ts = runBenchmark(X, JobQ, ResultQ, method=method, **kwargs)
            telapsed = time.time() - tstart

            Tinfo[method][rep] = dict()
            Tinfo[method][rep]['telapsed'] = telapsed
            Tinfo[method][rep]['ts'] = ts

    # PRINT RESULTS
    if 'monolithic' in Tinfo:
        telasped_monolithic = np.median([Tinfo['monolithic'][r]['telapsed']
                                         for r in range(nRepeat)])

    for rep in range(nRepeat):
        for method in methodNames:
            start, stop = [x for x in sliceGenerator(**kwargs)][0]
            msg = "%16s" % (method)
            if method == 'monolithic':
                msg += " %8d x %2d" % (X.shape[0], 1)
            else:
                msg += " %8d x %2d" % (stop - start, kwargs['nWorker'])
            msg += " %11.3f sec" % (np.median(Tinfo[method][rep]['ts']))

            telapsed = Tinfo[method][rep]['telapsed']
            msg += " %11.3f sec" % (telapsed)
            if 'monolithic' in Tinfo:
                msg += " %11.2f" % (telasped_monolithic / telapsed)
            print(msg)

    # PLOT RESULTS
    pylab.figure(1)
    pylab.hold('on')
    for method in methodNames:
        xs = kwargs['nWorker'] * np.ones(nRepeat)
        ys = [Tinfo[method][r]['telapsed'] for r in range(nRepeat)]
        if kwargs['nWorker'] == 1:
            label = method
        else:
            label = None
        pylab.plot(xs, ys, '.',
                   color=ColorMap[method],
                   markeredgecolor=ColorMap[method],
                   label=label)
    return Tinfo


class SharedMemWorker(multiprocessing.Process):

    """ Single "worker" process that processes tasks delivered via queues
    """

    def __init__(self, uid, JobQ, ResultQ, X):
        super(type(self), self).__init__()  # Required super constructor call
        self.uid = uid
        self.JobQ = JobQ
        self.ResultQ = ResultQ
        self.X = X

    def run(self):
        # Construct iterator with sentinel value of None (for termination)
        jobIterator = iter(self.JobQ.get, None)

        # Loop over tasks in the job queue
        for sliceArgs in jobIterator:
            start, stop, kwargs = sliceArgs
            t = workOnSlice(self.X, start, stop, **kwargs)
            self.ResultQ.put(t)
            self.JobQ.task_done()


def launchWorkers(X, nWorker=1, **kwargs):

    manager = multiprocessing.Manager()
    JobQ = manager.Queue()
    ResultQ = manager.Queue()

    for wID in range(nWorker):
        worker = SharedMemWorker(wID, JobQ, ResultQ, X)
        worker.start()
    return JobQ, ResultQ


def closeWorkers(JobQ, nWorker=1, **kwargs):
    for wID in range(nWorker):
        JobQ.put(None)  # this is shutdown signal


def makeData(N=10, D=10, **kwargs):
    PRNG = np.random.RandomState(0)
    X = PRNG.rand(N, D)
    return X


def problemGenerator(N=None, D=None, nWorker=None, **kwargs):
    iterator = itertools.product(
        rangeFromString(N),
        rangeFromString(D),
        rangeFromString(nWorker),
    )
    for N, D, nWorker in iterator:
        yield dict(N=N, D=D, nWorker=nWorker)


def makeTitle(N=0, D=0, nWorker=0,
              minSliceDuration=0,
              task='', memoryType='', scaleFactor=1.0, **kwargs):
    title = "N=%d D=%d nWorker=%d\n" \
        + "task %s\n" \
        + "minSliceDuration %s\n"\
        + "memoryType %s\n"\
        + "scaleFactor %s\n"
    return title % (N, D, nWorker, task,
                    minSliceDuration, memoryType, scaleFactor)


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


def sliceGenerator(N=0, nWorker=0, nTaskPerWorker=1, **kwargs):
    """ Iterate over slices given problem size and num workers

    Yields
    --------
    (start,stop) : tuple
    """
    sliceSize = np.floor(N / nWorker)
    for sliceID in range(nWorker * nTaskPerWorker):
        start = sliceID * sliceSize
        stop = (sliceID + 1) * sliceSize
        if sliceID == (nWorker * nTaskPerWorker) - 1:
            stop = N
        yield start, stop


if __name__ == "__main__":
    main()

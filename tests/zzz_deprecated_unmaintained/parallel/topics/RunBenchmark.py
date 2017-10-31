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
import bnpy
from bnpy.util import sharedMemDictToNumpy

ColorMap = dict(monolithic='k', serial='b', parallel='r')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('datasetName', type=str)
    parser.add_argument('--K', type=int, default=50)
    parser.add_argument('--nCoordAscentItersLP', type=int, default=100)
    parser.add_argument('--convThrLP', type=float, default=0.00001)
    parser.add_argument('--nWorker', type=str, default='1')
    parser.add_argument(
        '--maxWorker',
        type=int,
        default=multiprocessing.cpu_count())
    parser.add_argument('--methods', type=str, default='monolithic,parallel')
    parser.add_argument('--nRepeat', type=int, default=1)
    parser.add_argument('--task', type=str, default='localstep')
    parser.add_argument('--minSliceDuration', type=float, default=1.0)
    parser.add_argument('--scaleFactor', type=float, default=0.0)
    parser.add_argument('--memoryType', type=str, default='shared')
    parser.add_argument('--savefig', type=int, default=0)
    args = parser.parse_args()

    nWorker_IN = args.nWorker + ''
    pargs = args.__dict__
    pargs['methods'] = args.methods.split(',')

    for task in args.task.split(','):
        pargs['task'] = task
        AllInfo = initBenchmarkInfo(**pargs)
        pargs['scaleFactor'] = 0.0

        for nWorker in rangeFromString(nWorker_IN):
            # Make independent dict for all kwargs of current problem
            # so can edit these fields without editing the args for other tasks
            kwargs = copy.deepcopy(pargs)
            kwargs['nWorker'] = nWorker

            # Load the data
            Data = loadDataset(**kwargs)
            hmodel = createModel(Data, **kwargs)

            # Make shared memory
            dataSharedMem = Data.getRawDataAsSharedMemDict()
            aSharedMem = hmodel.allocModel.fillSharedMemDictForLocalStep()
            oSharedMem = hmodel.obsModel.fillSharedMemDictForLocalStep()
            ShMem = dict(dataSharedMem=dataSharedMem,
                         aSharedMem=aSharedMem,
                         oSharedMem=oSharedMem)

            # Make function handles
            fH = dict()
            fH['makeDataSliceFromSharedMem'] = Data.getDataSliceFunctionHandle(
            )
            fH['a_calcLocalParams'], fH['a_calcSummaryStats'] = \
                hmodel.allocModel.getLocalAndSummaryFunctionHandles()
            fH['o_calcLocalParams'], fH['o_calcSummaryStats'] = \
                hmodel.obsModel.getLocalAndSummaryFunctionHandles()

            # Finalize kwargs with model params
            aArgs = hmodel.allocModel.getSerializableParamsForLocalStep()
            oArgs = hmodel.obsModel.getSerializableParamsForLocalStep()
            kwargs.update(aArgs)
            kwargs.update(oArgs)
            kwargs['nDocTotal'] = Data.nDocTotal

            # Determine right scale of problem
            if np.allclose(pargs['scaleFactor'], 0) and args.task != 'sleep':
                pargs['scaleFactor'] = getScaleFactorForTask(
                    ShMem,
                    fH=fH,
                    **kwargs)
                kwargs['scaleFactor'] = pargs['scaleFactor']

            # Launch worker processes
            JobQ, ResultQ = launchWorkers(ShMem, fH, **kwargs)
            time.sleep(.1)

            CurInfo = defaultdict(dict)
            for rep in range(kwargs['nRepeat']):
                for method in getMethodNames(**kwargs):
                    tstart = time.time()
                    telapsed_slices = runBenchmark(
                        JobQ, ResultQ, ShMem,
                        method=method, fH=fH, **kwargs)
                    telapsed_wall = time.time() - tstart
                    CurInfo[method][rep] = dict()
                    CurInfo[method][rep]['telapsed_wall'] = telapsed_wall
                    CurInfo[method][rep]['telapsed_slices'] = telapsed_slices

                    print('%s %.2f ' % (method, telapsed_wall))

            AllInfo = updateBenchmarkInfo(AllInfo, CurInfo, **kwargs)
            printResultsTable(CurInfo, **kwargs)

            closeWorkers(JobQ, **kwargs)
            time.sleep(.1)

        plotSpeedupFigure(AllInfo, **kwargs)

        if not args.savefig:
            pylab.show(block=1)
        pylab.close('all')


def workOnSlice(ShMem,
                start=None, stop=None,
                fH=dict(),
                task='localstep',
                minSliceDuration=1.0,
                memoryType='shared',
                nWorker=1,
                maxWorker=1,
                scaleFactor=1.0,
                **kwargs):
    """ Perform work on a slice of data.
    """
    if start is not None:
        start = int(start)
        stop = int(stop)

    Dslice = fH['makeDataSliceFromSharedMem'](
        ShMem['dataSharedMem'], cslice=(start, stop))

    kwargs.update(sharedMemDictToNumpy(ShMem['aSharedMem']))
    kwargs.update(sharedMemDictToNumpy(ShMem['oSharedMem']))
    nReps = np.maximum(1, int(np.ceil(scaleFactor)))
    tstart = time.time()
    if task == 'sleep':
        # Make sure that we sleep same total amt whether serial or parallel
        duration = minSliceDuration * (maxWorker / float(nWorker))
        time.sleep(duration)
    elif task == 'localstep':
        for rep in range(nReps):
            # Do local step
            LP = fH['o_calcLocalParams'](Dslice, **kwargs)
            LP = fH['a_calcLocalParams'](Dslice, LP, **kwargs)

            # Do summary step
            SS = fH['a_calcSummaryStats'](
                Dslice, LP, doPrecompEntropy=1, **kwargs)
            SS = fH['o_calcSummaryStats'](Dslice, SS, LP, **kwargs)

    else:
        raise NotImplementedError("Unrecognized task: %s" % (task))
    telapsed = time.time() - tstart
    return telapsed


class SharedMemWorker(multiprocessing.Process):

    """ Single "worker" process that processes tasks delivered via queues
    """

    def __init__(self, uid, JobQ, ResultQ, ShMem, fH, **kwargs):
        super(type(self), self).__init__()  # Required super constructor call
        self.uid = uid
        self.JobQ = JobQ
        self.ResultQ = ResultQ
        self.ShMem = ShMem
        self.fH = fH
        self.kwargs = kwargs

    def run(self):
        # Construct iterator with sentinel value of None (for termination)
        jobIterator = iter(self.JobQ.get, None)

        for start, stop in jobIterator:
            t = workOnSlice(self.ShMem,
                            start=start, stop=stop,
                            fH=self.fH, **self.kwargs)
            self.ResultQ.put(t)
            self.JobQ.task_done()


def launchWorkers(ShMem, fH, nWorker=1, **kwargs):

    manager = multiprocessing.Manager()
    JobQ = manager.Queue()
    ResultQ = manager.Queue()

    for wID in range(nWorker):
        worker = SharedMemWorker(wID, JobQ, ResultQ, ShMem, fH, **kwargs)
        worker.start()
    return JobQ, ResultQ


def closeWorkers(JobQ, nWorker=1, **kwargs):
    for wID in range(nWorker):
        JobQ.put(None)  # this is shutdown signal


def loadDataset(datasetName='', **kwargs):
    sys.path.append(os.environ['BNPYDATADIR'])
    DataMod = __import__(datasetName, fromlist=[])
    Data = DataMod.get_data(**kwargs)
    return Data


def createModel(Data, K=50, **kwargs):
    hmodel = bnpy.HModel.CreateEntireModel(
        'VB', 'FiniteTopicModel', 'Mult',
        dict(alpha=0.1, gamma=5),
        dict(lam=0.1),
        Data)
    hmodel.init_global_params(Data, initname='randexamples', K=K)
    return hmodel


def getMethodNames(methods='all', **kwargs):
    allMethodNames = ['monolithic', 'serial', 'parallel']
    methodNames = list()
    for name in allMethodNames:
        if 'all' not in methods and name not in methods:
            continue
        methodNames.append(name)
    return methodNames


def printResultsTable(Tinfo, nRepeat=1, methods='', **kwargs):

    print('======================= ', makeTitle(**kwargs))
    print('%16s %15s %15s %15s %10s' % (
        ' ', 'slice size', 'slice time', 'wallclock time', 'speedup'))

    nDocTotal = kwargs['nDocTotal']
    # PRINT RESULTS
    if 'monolithic' in Tinfo:
        telasped_monolithic = np.median(
            [Tinfo['monolithic'][r]['telapsed_wall'] for r in range(nRepeat)])

    for rep in range(nRepeat):
        for method in getMethodNames(methods):
            start, stop = [x for x in sliceGenerator(**kwargs)][0]
            msg = "%16s" % (method)
            if method == 'monolithic':
                msg += " %8d x %2d" % (nDocTotal, 1)
            else:
                msg += " %8d x %2d" % (stop - start, kwargs['nWorker'])
            msg += " %11.3f sec" % (
                np.median(Tinfo[method][rep]['telapsed_slices']))

            telapsed = Tinfo[method][rep]['telapsed_wall']
            msg += " %11.3f sec" % (telapsed)
            if 'monolithic' in Tinfo:
                msg += " %11.2f" % (telasped_monolithic / telapsed)
            print(msg)


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

    if kwargs['savefig']:
        title = 'BenchmarkPlot_%s_%s_minDur=%.2f_Speedup.eps'\
            % (platform.node(), kwargs['task'], kwargs['minSliceDuration'])
        pylab.savefig(title,
                      format='eps',
                      bbox_inches='tight',
                      pad_inches=0)


def getScaleFactorForTask(
        ShMem, fH=dict(), maxWorker=1, nDocTotal=0, **kwargs):

    minSliceDuration = kwargs['minSliceDuration']
    sliceSize = np.floor(nDocTotal / maxWorker)

    kwargs['scaleFactor'] = 1
    print('FINDING PROBLEM SCALE WITH SPECIFIED DURATION...')
    print('  Max possible workers: %d\n  Min duration of slice: %.2f' \
        % (maxWorker, minSliceDuration))

    t = workOnSlice(ShMem, 0, sliceSize, fH=fH, **kwargs)
    while t < minSliceDuration:
        kwargs['scaleFactor'] *= 2
        t = workOnSlice(ShMem, 0, sliceSize, fH=fH, **kwargs)
    print('FINAL SCALE:')
    print('  nReps: ', kwargs['scaleFactor'])
    print('  Slice Duration: %.2f sec' % (t))
    kwargs['minSliceDuration'] = t
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
            CurInfo[method][r]['telapsed_wall'] for r in range(nRepeat)])
        key = 't_' + method
        loc = np.flatnonzero(AllInfo['nWorker'] == kwargs['nWorker'])
        AllInfo[key][loc] = np.median(tvec)
    return AllInfo


def runBenchmark(
        JobQ, ResultQ,
        ShMem,
        method='serial',
        nWorker=1, nTaskPerWorker=1,
        **kwargs):
    """ Run benchmark timing experiment for specific computation method.
    """
    nDoc = kwargs['nDocTotal']
    ts = list()
    if method == 'monolithic':
        t = workOnSlice(ShMem, None, None, **kwargs)
        ts.append(t)
    elif method == 'serial':
        kwargs['nWorker'] = nWorker  # scale work load by num workers
        for start, stop in sliceGenerator(nDoc, nWorker, nTaskPerWorker):
            t = workOnSlice(ShMem, start, stop, **kwargs)
            ts.append(t)
    elif method == 'parallel':
        kwargs['nWorker'] = nWorker  # scale work load by num workers
        for start, stop in sliceGenerator(nDoc, nWorker, nTaskPerWorker):
            JobQ.put((start, stop))
        JobQ.join()  # requires each worker to use task_done to unblock
        while not ResultQ.empty():
            t = ResultQ.get()
            ts.append(t)
    return ts


def makeTitle(nDocTotal=0, nWorker=0,
              minSliceDuration=0,
              task='', memoryType='', scaleFactor=1.0, **kwargs):
    title = "nDoc=%d nWorker=%d\n" \
        + "task %s\n" \
        + "minSliceDuration %s\n"\
        + "memoryType %s\n"\
        + "scaleFactor %s\n"
    return title % (nDocTotal, nWorker, task,
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


def sliceGenerator(nDocTotal=0, nWorker=0, nTaskPerWorker=1, **kwargs):
    """ Iterate over slices given problem size and num workers

    Yields
    --------
    (start,stop) : tuple
    """
    sliceSize = np.floor(nDocTotal / nWorker)
    for sliceID in range(nWorker * nTaskPerWorker):
        start = sliceID * sliceSize
        stop = (sliceID + 1) * sliceSize
        if sliceID == (nWorker * nTaskPerWorker) - 1:
            stop = nDocTotal
        yield start, stop


if __name__ == "__main__":
    main()

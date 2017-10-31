"""
Shared memory parallel implementation of LP local step

Classes
--------

SharedMemWorker : subclass of Process
    Defines work to be done by a single "worker" process
    which is created with references to shared read-only data
    We assign this process "jobs" via a queue, and read its results
    from a separate results queue.

"""

import sys
import os
import multiprocessing
from multiprocessing import sharedctypes
import itertools
import numpy as np
import ctypes
import time

import bnpy

from bnpy.util import sharedMemDictToNumpy, sharedMemToNumpyArray
from bnpy.data.DataIteratorFromDisk import loadDataForSlice


class SharedMemWorker(multiprocessing.Process):

    """ Single "worker" process that processes tasks delivered via queues
    """

    def __init__(self, uid, JobQueue, ResultQueue,
                 makeDataSliceFromSharedMem,
                 o_calcLocalParams,
                 o_calcSummaryStats,
                 a_calcLocalParams,
                 a_calcSummaryStats,
                 dataSharedMem,
                 aSharedMem,
                 oSharedMem,
                 LPkwargs=None,
                 verbose=0):
        super(SharedMemWorker, self).__init__()
        self.uid = uid
        self.JobQueue = JobQueue
        self.ResultQueue = ResultQueue

        # Function handles
        self.makeDataSliceFromSharedMem = makeDataSliceFromSharedMem
        self.o_calcLocalParams = o_calcLocalParams
        self.o_calcSummaryStats = o_calcSummaryStats
        self.a_calcLocalParams = a_calcLocalParams
        self.a_calcSummaryStats = a_calcSummaryStats

        # Things to unpack
        self.dataSharedMem = dataSharedMem
        self.aSharedMem = aSharedMem
        self.oSharedMem = oSharedMem
        if LPkwargs is None:
            LPkwargs = dict()
        self.LPkwargs = LPkwargs

        self.verbose = verbose

    def printMsg(self, msg):
        if self.verbose:
            for line in msg.split("\n"):
                print("#%d: %s" % (self.uid, line))

    def run(self):
        self.printMsg("process SetUp! pid=%d" % (os.getpid()))
        # Construct iterator with sentinel value of None (for termination)
        jobIterator = iter(self.JobQueue.get, None)

        for sliceArgs, aArgs, oArgs in jobIterator:
            start1 = time.time()
            # Grab slice of data to work on
            if 'start' in sliceArgs:
                # Load data from shared memory
                Dslice = self.makeDataSliceFromSharedMem(
                    self.dataSharedMem,
                    batchID=sliceArgs['batchID'],
                    cslice=(sliceArgs['start'], sliceArgs['stop']))
            else:
                # Load data slice directly from file
                Dslice = loadDataForSlice(**sliceArgs)

            # Prep kwargs for the alloc model, especially local step kwargs
            aArgs.update(sharedMemDictToNumpy(self.aSharedMem))
            aArgs.update(self.LPkwargs)
            aArgs.update(sliceArgs)

            # Prep kwargs for the obs model
            oArgs.update(sharedMemDictToNumpy(self.oSharedMem))

            # Do local step
            LP = self.o_calcLocalParams(Dslice, **oArgs)
            LP = self.a_calcLocalParams(Dslice, LP, **aArgs)

            # Do summary step
            SS = self.a_calcSummaryStats(
                Dslice, LP, doPrecompEntropy=1, **aArgs)
            SS = self.o_calcSummaryStats(Dslice, SS, LP, **oArgs)

            # Add final suff stats to result queue to wrap up this task!
            self.ResultQueue.put(SS)
            self.JobQueue.task_done()

            duration = time.time() - start1
        # Clean up
        self.printMsg("process CleanUp! pid=%d" % (os.getpid()))

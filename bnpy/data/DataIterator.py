'''
DataIterator.py

Object that manages iterating over minibatches of a dataset.

Usage
--------
Construct by providing the underlying full-dataset,
as an object of type bnpy.data.DataObj
>> I = DataIterator(Data, nBatch=10, nLap=3)

To determine if more data remains, call *has_next_batch*
>> I.has_next_batch()

To access the next batch, call the *get_next_batch* method.
This returns a DataObj dataset, of same type as Data
>> DataChunk = I.get_next_batch()

Notes
------
The subsets or batches are defined by the constructor,
via a random partition of all data units.
* For "flat" datasets, each unit is a single observation.
* For document-structured datasets, each unit is one document.
* For sequence datasets: each unit is a full sequence.

For example, given 10 documents, a possible set of 3 batches is
   batch 1 : docs 1, 3, 9, 4,
   batch 2 : docs 5, 7, 0
   batch 3 : docs 8, 2, 6
Each lap (pass through the data) iterates through these same fixed batches.

The traversal order of the batch is randomized at each lap.
For example, during the first 3 laps, we may see the following orders
   lap 0 : batches 0, 2, 1
   lap 1 : batches 2, 1, 0
   lap 2 : batches 0, 1, 2
Set the "dataorderseed" parameter to get repeatable orders.

'''

import numpy as np
MAXSEED = 1000000


class DataIterator(object):

    """ Object that manages iterating over minibatches of a dataset.

    Methods
    ------
    get_next_batch() : get next minibatch of this dataset.

    Attributes
    ------
    DataPerBatch : dict
        key/value pairs map batchIDs to dataset objects
    nBatch : int
        total number of batches provided dataset is divided into
    nLap : int
        number of laps (passes thru whole dataset) to complete
    batchID : int
        integer ID of the most recent batch returned by get_next_batch()
        batchID has range [0, nBatch-1]
    curLapPos : int
        integer ID of current position in batch order.
        Range is [0, nBatch-1].
        curLapPos is always incremented by 1 after every
        call to get_next_batch()
    lapID : int
        integer ID of the current lap.
        Range is [0, nLap-1].
        lapID is always incremented by one after each lap.
    """

    def __init__(self, Data, nBatch=10, nLap=10,
                 dataorderseed=42, startLap=0,
                 alwaysTrackTruth=False, **kwargs):
        ''' Create an iterator over batches/subsets of a large dataset.

        Each batch/subset is represented by an instance of
        a bnpy.data.DataObj object. Each such batch-specific object is
        configured so that it is aware of the total size of the
        whole dataset.

        Parameters
        ------
        Data : bnpy.data.DataObj
        nBatch : int
            total number of batches provided dataset is divided into
        nLap : int
            number of laps (passes thru whole dataset) to complete
        dataorderseed : int
            seed for random number generator that determines
            random division of data into fixed set of batches
            and random order for visiting batches during each lap
        '''
        self.Data = Data

        if hasattr(Data, 'name'):
            self.name = Data.name

        nBatch = int(np.minimum(nBatch, Data.get_total_size()))
        self.nBatch = nBatch
        # TODO: Warn about using fewer than specified num batches

        self.nLap = nLap + int(startLap)

        # Config order in which batches are traversed
        self.curLapPos = -1
        self.lapID = int(startLap)
        self.dataorderseed = int(int(dataorderseed) % MAXSEED)

        # Decide how many units will be in each batch
        # nUnitPerBatch : 1D array, size nBatch
        # nUnitPerBatch[b] gives total number of units in batch b
        nUnit = Data.get_size()
        nUnitPerBatch = nUnit // nBatch * np.ones(nBatch, dtype=np.int32)
        nRem = nUnit - nUnitPerBatch.sum()
        nUnitPerBatch[:nRem] += 1

        # Randomly assign each unit to exactly one batch
        PRNG = np.random.RandomState(self.dataorderseed)
        shuffleIDs = PRNG.permutation(nUnit).tolist()
        self.DataPerBatch = list()
        self.IDsPerBatch = list()
        for b in xrange(nBatch):
            curBatchMask = shuffleIDs[:nUnitPerBatch[b]]
            Dchunk = Data.make_subset(
                curBatchMask, doTrackTruth=alwaysTrackTruth)
            Dchunk.alwaysTrackTruth = alwaysTrackTruth
            self.DataPerBatch.append(Dchunk)
            self.IDsPerBatch.append(curBatchMask)
            # Remove units assigned to this batch
            # from consideration for future batches
            del shuffleIDs[:nUnitPerBatch[b]]

        # Decide which order the batches will be traversed in the first lap
        self.batchOrderCurLap = self.getRandPermOfBatchIDsForCurLap()

    def has_next_batch(self):
        if self.lapID >= self.nLap:
            return False
        if self.lapID == self.nLap - 1:
            if self.curLapPos == self.nBatch - 1:
                return False
        return True

    def get_next_batch(self, batchIDOnly=False):
        ''' Get the Data object for the next batch.

        Raises
        --------
        StopIteration if we have completed all specified laps

        Post Condition
        --------
        - batchID equals index of most recent batch returned.
        - lapID equals how many laps have been *completed*.
        - curLapPos int indicating progress through current lap.
            if we are on first batch, curLapPos=0
            if we are b-th batch, curLapPos=b-1
            if we are on last batch, curLapPos=nBatch-1

        Returns
        --------
        Data : bnpy Data object for the current batch
        '''
        if not self.has_next_batch():
            raise StopIteration()

        self.curLapPos += 1
        if self.curLapPos >= self.nBatch:
            # Starting a new lap!
            self.curLapPos = 0
            self.lapID += 1
            self.batchOrderCurLap = self.getRandPermOfBatchIDsForCurLap()

        # Create the DataObj for the current batch
        self.batchID = self.batchOrderCurLap[self.curLapPos]
        if batchIDOnly:
            return self.batchID
        else:
            return self.DataPerBatch[self.batchID]

    def getRandPermOfBatchIDsForCurLap(self):
        ''' Returns array of batchIDs, permuted in random order.

        Random seed used for permutation is determined by:
            seed = dataorderseed + lapID

        This allows us to always jump to lap L
        and get reproduce its order exactly.

        Returns
        -------
        curBatchIDs : 1D array, size nBatch
            random permutation of integers [0, 1, ... nBatch-1]
        '''
        curseed = int(self.dataorderseed + self.lapID)
        PRNG = np.random.RandomState(curseed)
        return PRNG.permutation(self.nBatch)

    def get_stats_summary(self):
        ''' Returns human-readable summary of this dataset's basic properties

        Returns
        ------
        s : string
        '''
        nPerBatch = self.DataPerBatch[0].get_size()
        s = '  total size: %d units\n' % (self.Data.get_total_size())
        s += '  batch size: %d units\n' % (nPerBatch)
        s += '  num. batches: %d' % (self.nBatch)
        return s

    def get_text_summary(self):
        ''' Returns human-readable one-line description of this dataset.

        Returns
        ------
        s : string
        '''
        return self.Data.get_text_summary()

    def getBatch(self, batchID):
        ''' Returns Data object for requested batch

        Returns
        -------
        Dbatch : bnpy DataObj
        '''
        return self.DataPerBatch[batchID]

    def getRawDataAsSharedMemDict(self):
        ''' Create dict with copies of raw data as shared memory arrays
        '''
        dataShMemDict = dict()
        for batchID in xrange(self.nBatch):
            BatchData = self.DataPerBatch[batchID]
            ShMem = self.DataPerBatch[batchID].getRawDataAsSharedMemDict()
            dataShMemDict[batchID] = ShMem
        return dataShMemDict

    def getDataSliceFunctionHandle(self):
        """ Return function handle that can make data slice objects.

        Useful with parallelized algorithms,
        when we need to use shared memory.

        Returns
        -------
        f : function handle
        """
        return self.DataPerBatch[0].getDataSliceFunctionHandle()

    def calcSliceArgs(self, batchID, workerID, nWorkers, lapFrac):
        nUnits = self.DataPerBatch[batchID].get_size()
        nUnitsPerSlice = int(np.floor(nUnits / nWorkers))
        start = workerID * nUnitsPerSlice
        if workerID == nWorkers - 1:
            stop = nUnits
        else:
            stop = (workerID + 1) * nUnitsPerSlice
        SliceInfo = dict(batchID=batchID, start=start, stop=stop,
                         lapFrac=lapFrac, sliceID=workerID)
        return SliceInfo

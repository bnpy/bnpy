'''
FromLP.py

Initialize global params of a bnpy model using a set of local parameters
'''
from builtins import *
import numpy as np

from .FromTruth import convertLPFromHardToSoft

import logging
Log = logging.getLogger('bnpy')
Log.setLevel(logging.DEBUG)


def init_global_params(hmodel, Data, initname='', initLP=None,
                       **kwargs):
    ''' Initialize (in-place) the global params of the given hmodel.

    Parameters
    -------
    hmodel : bnpy.HModel
        model object to initialize
    Data   : bnpy.data.DataObj
         Dataset to use to drive initialization.
         hmodel.obsModel dimensions must match this dataset.
    initname : str, ['contigblocksLP', 'sacbLP']
        name for the routine to use

    Post Condition
    --------
    hmodel has valid global parameters.
    '''
    if isinstance(initLP, dict):
        return initHModelFromLP(hmodel, Data, initLP)

    elif initname == 'sacbLP':
        Log.info('Initialization: Sequential Allocation of Contig Blocks')
        SS = initSS_SeqAllocContigBlocks(Data, hmodel, **kwargs)
        hmodel.update_global_params(SS)
        return None

    elif initname == 'contigblocksLP':
        LP = makeLP_ContigBlocks(Data, **kwargs)
        return initHModelFromLP(hmodel, Data, LP)

    else:
        raise ValueError('Unrecognized initname: %s' % (initname))


def initHModelFromLP(hmodel, Data, LP):
    ''' Initialize provided bnpy HModel given data and local params.

    Executes summary step and global step given the provided LP.

    Post Condition
    ------
    hmodel has valid global parameters.
    '''
    if 'resp' not in LP:
        if 'Z' not in LP:
            raise ValueError("Bad LP. Require either 'resp' or 'Z' fields.")
        LP = convertLPFromHardToSoft(LP, Data)
    assert 'resp' in LP

    if hasattr(hmodel.allocModel, 'initLPFromResp'):
        LP = hmodel.allocModel.initLPFromResp(Data, LP)
    SS = hmodel.get_global_suff_stats(Data, LP)
    hmodel.update_global_params(SS)


def makeLP_ContigBlocks(Data, K=0, KperSeq=None, initNumSeq=None, **kwargs):
    ''' Create local parameters via a contiguous block hard segmentation.

    Divide chosen sequences up into KperSeq contiguous blocks,
    each block evenly sized, and assign each block to a unique state.

    Returns
    -------
    LP : dict of local parameters
        * resp : 2D array, Natom x K
    '''
    if initNumSeq is None:
        initNumSeq = Data.nDoc
    initNumSeq = np.minimum(initNumSeq, Data.nDoc)

    if KperSeq is None:
        assert K > 0
        KperSeq = int(np.ceil(K / float(initNumSeq)))
        if KperSeq * initNumSeq > K:
            print('WARNING: using initial K larger than suggested.')
        K = KperSeq * initNumSeq
    assert KperSeq > 0

    # Select subset of all sequences to use for initialization
    if initNumSeq == Data.nDoc:
        chosenSeqIDs = np.arange(initNumSeq)
    else:
        chosenSeqIDs = PRNG.choice(Data.nDoc, initNumSeq, replace=False)

    # Make hard segmentation at each chosen sequence
    resp = np.zeros((Data.nObs, K))
    jstart = 0
    for n in chosenSeqIDs:
        start = int(Data.doc_range[n])
        curT = Data.doc_range[n + 1] - start

        # Determine how long each block is for blocks 0, 1, ... KperSeq-1
        cumsumBlockSizes = calcBlockSizesForCurSeq(KperSeq, curT)
        for j in range(KperSeq):
            Tstart = start + cumsumBlockSizes[j]
            Tend = start + cumsumBlockSizes[j + 1]
            resp[Tstart:Tend, jstart + j] = 1.0
        jstart = jstart + j + 1

    return dict(resp=resp)


def calcBlockSizesForCurSeq(KperSeq, curT):
    ''' Divide a sequence of length curT into KperSeq contig blocks

        Examples
        ---------
        >> calcBlockSizesForCurSeq(3, 20)
        [0, 7, 14, 20]

        Returns
        ---------
        c : 1D array, size KperSeq+1
        * block t indices are selected by c[t]:c[t+1]
    '''
    blockSizes = (curT // KperSeq) * np.ones(KperSeq)
    remMass = curT - np.sum(blockSizes)
    blockSizes[:remMass] += 1
    cumsumBlockSizes = np.cumsum(np.hstack([0, blockSizes]))
    return np.asarray(cumsumBlockSizes, dtype=np.int32)


def initSS_SeqAllocContigBlocks(Data, hmodel, **kwargs):

    if 'seed' in kwargs:
        seed = int(kwargs['seed'])
    else:
        seed = 0
    # Traverse sequences in a random order
    PRNG = np.random.RandomState(seed)
    assert hasattr(Data, 'nDoc')
    randOrderIDs = list(range(Data.nDoc))
    PRNG.shuffle(randOrderIDs)

    SS = None
    for orderID, n in enumerate(randOrderIDs):
        hmodel, SS = initSingleSeq_SeqAllocContigBlocks(
            n, Data, hmodel,
            SS=SS,
            **kwargs)
        if orderID == len(randOrderIDs) - 1 \
           or (orderID + 1) % 5 == 0 or orderID < 2:
            Log.info('  seq. %3d/%d | Ktotal=%d'
                     % (orderID + 1, len(randOrderIDs), SS.K))
    return SS

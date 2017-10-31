'''
FromTruth.py

Initialize params of a bnpy model using "ground truth" information,
such as human annotations

These are provided within the Data object's TrueParams attribute.
'''
from __future__ import print_function
from builtins import *
import numpy as np


def init_global_params(hmodel, Data, initname=None, seed=0, **kwargs):
    ''' Initialize global params of hmodel using Data's ground truth.

    Parameters
    -------
    hmodel : bnpy.HModel
        Model object to initialize.
    Data   : bnpy.data.DataObj
        Dataset to use to drive initialization.
        obsModel dimensions must match this dataset.
    initname : str
        name of routine used to do initialization
        Options: ['trueparams', 'truelabels',
                  'repeattruelabels',
                  'truelabelsandempties',
                  'truelabelsandjunk',
                 ]

    Post Condition
    -------
    hmodel object has valid global parameters,
    for both its allocModel and its obsModel.
    '''
    PRNG = np.random.RandomState(seed)
    if initname.count('truelabels') > 0:
        _initFromTrueLP(hmodel, Data, initname, PRNG, **kwargs)
    elif initname.count('trueparams') > 0:
        _initFromTrueParams(hmodel, Data, initname, PRNG, **kwargs)
    else:
        raise NotImplementedError('Unknown initname: %s' % (initname))

    if hmodel.obsModel.inferType == 'EM':
        assert hasattr(hmodel.obsModel, 'EstParams')
    else:
        assert hasattr(hmodel.obsModel, 'Post')


def _initFromTrueParams(hmodel, Data, initname, PRNG, **kwargs):
    ''' Initialize global parameters of provided model to specific values

    Uses named parameters in the dataset's TrueParams attribute,
    and passes these as kwargs to the set_global_params methods
    implemented by the allocModel and obsModel.

    Post Condition
    -------
    hmodel object has valid global parameters,
    for both its allocModel and its obsModel.
    '''
    InitParams = dict(**Data.TrueParams)
    InitParams['Data'] = Data
    hmodel.set_global_params(**InitParams)


def _initFromTrueLP(hmodel, Data, initname, PRNG, nRepeatTrue=2,
                    initKextra=1,
                    **kwargs):
    ''' Initialize global params of provided model given local assignments.

    Uses the 'Z' or 'resp' fields of the dataset's TrueParams dict
    to create a local parameters dictionary. This is then used to do a
    summary step (call get_global_suff_stats) and then a global step
    (call update_global_params).

    Post Condition
    -------
    hmodel object has valid global parameters,
    for both its allocModel and its obsModel.
    '''

    # Extract "true" local params dictionary LP specified in the Data struct
    LP = dict()
    if hasattr(Data, 'TrueParams') and 'Z' in Data.TrueParams:
        LP['Z'] = Data.TrueParams['Z']
        LP = convertLPFromHardToSoft(LP, Data)
    elif hasattr(Data, 'TrueParams') and 'resp' in Data.TrueParams:
        LP['resp'] = Data.TrueParams['resp']
    else:
        raise ValueError(
            'init_global_params requires TrueLabels or TrueParams.')

    # Convert between token/doc responsibilities
    if str(type(hmodel.obsModel)).count('Mult'):
        if hmodel.obsModel.DataAtomType == 'doc':
            LP = convertLPFromTokensToDocs(LP, Data)
        else:
            LP = convertLPFromDocsToTokens(LP, Data)
    elif str(type(hmodel.obsModel)).count('Bern'):
        if hmodel.obsModel.DataAtomType == 'doc':
            LP = convertLPFromTokensToDocs(LP, Data)
        elif hasattr(Data, 'word_count'):
            LP = convertLPFromTokensToTypes(LP, Data)
        else:
            # Leave Gaussian stuff alone
            pass

    # Adjust "true" labels as specified by initname
    if initname == 'repeattruelabels':
        LP = expandLPWithDuplicates(LP, PRNG, nRepeatTrue)
    elif initname == 'subdividetruelabels':
        LP = expandLPWithContigBlocks(LP, Data, PRNG)
    elif initname.count('empty') or initname.count('empties'):
        LP = expandLPWithEmpty(LP, initKextra)
    elif initname.count('junk'):
        LP = expandLPWithJunk(LP, initKextra, PRNG=PRNG, **kwargs)
    elif initname.count('dropwords'):
        LP = dropWordsFromLP(Data, LP, PRNG=PRNG, **kwargs)

    if hasattr(hmodel.allocModel, 'initLPFromResp'):
        LP = hmodel.allocModel.initLPFromResp(Data, LP)

    # Perform global update step given these local params
    SS = hmodel.get_global_suff_stats(Data, LP)
    hmodel.update_global_params(SS)


def convertLPFromHardToSoft(LP, Data,
                            initGarbageState=1,
                            returnZ=False,
                            startIDsAt0=False,
                            Kmax=None):
    ''' Transform array of hard assignment labels in Data into local param dict

    Keyword Args
    ------------
    initGarbageState : integer flag (0/1)
        if on, will add a garbage state for each negative id in TrueZ
    startIDsAt0 : integer flag (0/1)
        if off, will index states from 0, 1, ... Kmax-1 with no skipping.
        if on, can potentially have some states with no assigned data.

    Returns
    ---------
    LP : dict
        with updated fields
        * 'resp' : 2D array, N x K
    '''
    Z = LP['Z']
    uniqueLabels = np.unique(Z)
    uniqueAssigned = [u for u in uniqueLabels if u >= 0]
    if startIDsAt0:
        if Kmax is None:
            Kmax = np.max(uniqueAssigned) + 1
        uniqueAssigned = np.arange(Kmax)
    else:
        Kmax = len(uniqueAssigned)

    if hasattr(Data, 'edges'):
        if Z.ndim == 3:
            # Unpack adjacency-matrix formatted Z
            Zsrc = np.zeros(Data.nEdges, dtype=np.int32)
            Zrcv = np.zeros(Data.nEdges, dtype=np.int32)
            for eid, (s, t) in enumerate(Data.edges):
                Zsrc[eid] = Z[s, t, 0]
                Zrcv[eid] = Z[s, t, 1]
        else:
            Zsrc = Z[:,0]
            Zrcv = Z[:,1]
        resp = np.zeros((Zsrc.size, Kmax, Kmax))
        for eid in range(Data.nEdges):
            resp[eid, Zsrc[eid], Zrcv[eid]] = 1.0
        LP['resp'] = resp
        return LP

    garbageMask = Z < 0
    if np.sum(garbageMask) > 0 and initGarbageState:
        Kgarbage = np.unique(Z[garbageMask]).size
        resp = np.zeros((Z.size, Kmax + Kgarbage))
        for kk in range(Kgarbage):
            resp[Z == -1 - kk, Kmax + kk] = 1
    else:
        resp = np.zeros((Z.size, Kmax))

    # Fill in "real" states
    for k in range(Kmax):
        mask = Z == uniqueAssigned[k]
        resp[mask, k] = 1.0
    LP['resp'] = resp

    if returnZ:
        Znew = -1 * np.zeros_like(Z)
        for k in range(Kmax):
            mask = Z == uniqueAssigned[k]
            Znew[mask] = k
        assert np.all(Znew >= 0)
        return LP, Znew
    return LP


def expandLPWithEmpty(LP, Kextra):
    ''' Create new LP by adding empty columns at the end

    Parameters
    --------
    LP : dict
        local parameters dict with K components
    Kextra : int
        number of new components to insert

    Returns
    --------
    LP : dict,
        with K + Kextra total components.
    '''
    resp = LP['resp']
    LP['resp'] = np.hstack([resp, np.zeros((resp.shape[0], Kextra))])
    return LP

def dropWordsFromLP(Data, LP,
                    PRNG=np.random,
                    initTargetCompID=0,
                    initDropWordIDs=None,
                    initNumDropWords=10,
                    **kwargs):
    ''' Select a topic and remove certain words from its support entirely.

    Returns
    --------
    LP : dict,
        with K + Kextra total components.
    '''
    resp = LP['resp']
    Mat = Data.getSparseTokenTypeCountMatrix()
    wordCount_k = Mat * resp[:, initTargetCompID]

    if initDropWordIDs is None or initDropWordIDs == 'None':
        initDropWordIDs = np.argsort(-1 * wordCount_k)[:initNumDropWords]
    if isinstance(initDropWordIDs, str):
        initDropWordIDs = [int(x) for x in initDropWordIDs.split(',')]

    initDropWordIDs = np.asarray(initDropWordIDs)
    for v in initDropWordIDs:
        mask = Data.word_id == v
        resp[mask, initTargetCompID] = 1e-100

    print('initDropWordIDs:', end=' ')
    print('   ', initDropWordIDs)

    LP['resp'] = resp
    return LP

def expandLPWithJunk(LP, Kextra, PRNG=np.random.RandomState,
        initFracJunk=0.01, **kwargs):
    ''' Create new LP by adding extra junk topics

    Parameters
    --------
    LP : dict
        local parameters dict with K components
    Kextra : int
        number of new components to insert

    Returns
    --------
    LP : dict,
        with K + Kextra total components.
    '''
    resp = LP['resp']
    N, K = resp.shape
    respNew = np.hstack([resp, np.zeros((N, Kextra))])
    Nextra = int(initFracJunk * N)
    selectIDs = PRNG.choice(N, Nextra * Kextra).tolist()
    for k in range(Kextra):
        IDs_k = selectIDs[:Nextra]
        respNew[IDs_k, :K] = 0.01 / K
        respNew[IDs_k, K + k] = 1 - 0.01
        del selectIDs[:Nextra]
    return dict(resp=respNew)


def expandLPWithDuplicates(LP, PRNG, nRepeatTrue=2):
    ''' Create new LP by randomly splitting each existing component.

    Creates nRepeatTrue "near-duplicates" of each comp,
    with each original member of comp k assigned to one of these
    duplicates at random.

    Parameters
    --------
    LP : dict
        local parameters dict with K components
    PRNG : random number generator
    nRepeatTrue : int
        number of states to split each original component into

    Returns
    --------
    LP : dict,
        with K*nRepeatTrue total components.
    '''
    resp = LP['resp']
    N, Ktrue = resp.shape
    rowIDs = PRNG.permutation(N)
    L = len(rowIDs) / nRepeatTrue
    bigResp = np.zeros((N, Ktrue * nRepeatTrue))
    curLoc = 0
    for r in range(nRepeatTrue):
        targetIDs = rowIDs[curLoc:curLoc + L]
        bigResp[targetIDs, r * Ktrue:(r + 1) * Ktrue] = resp[targetIDs, :]
        curLoc += L
    LP['resp'] = bigResp
    return LP


def expandLPWithContigBlocks(LP, Data, PRNG, nPerSeq=2,
                             numNewOptions=[2, 3, 4, 5],
                             pNumNew=[0.45, 0.45, 0.05, 0.05],
                             Kmax=64):
    ''' Expand hard labels at randomly-chosen contiguous blocks

    Example
    -------
    [0, 0, 0, 0,    1, 1, 1, 1, 1,   2, 2, 2, 2, 2]
    could become
    [3, 3, 4, 4,    1, 1, 1, 1, 1,   5, 5, 6, 6, 6]

    Returns
    ------
    LP : dict
    '''
    Z = LP['Z']
    knewID = Z.max() + 1
    for n in range(Data.nDoc):
        start = Data.doc_range[n]
        stop = Data.doc_range[n + 1]
        Z_n = Z[start:stop]
        breakLocs = np.flatnonzero(Z_n[:-1] - Z_n[1:]) + 1
        breakLocs = np.hstack([0, breakLocs, Z_n.size])
        assert np.all(np.diff(breakLocs) > 0)
        # Select nPerSeq breakpoints at random
        L = len(breakLocs) - 1
        nSamps = np.minimum(L, nPerSeq)
        chosenIDs = PRNG.choice(L, nSamps, replace=False)
        for cc in chosenIDs:
            start = breakLocs[cc]
            stop = breakLocs[cc + 1]
            nNew = PRNG.choice(numNewOptions, p=pNumNew)
            nNew = np.minimum(nNew, stop - start)
            Bs = (stop - start) / nNew * np.ones(nNew, dtype=np.int32)
            gap = (stop - start) - Bs.sum()
            if gap > 0:
                Bs[:gap] += 1
            for nn in range(nNew):
                if nn == 0:
                    a = start
                Z_n[a:a + Bs[nn]] = knewID
                a = a + Bs[nn]
                knewID += 1
                if knewID >= Kmax:
                    break  # exceed max capacity of number of comps
            if knewID >= Kmax:
                break  # exceed max capacity of number of comps
        if knewID >= Kmax:
            break  # exceed max capacity of number of comps

    return convertLPFromHardToSoft(LP, Data)

def convertLPFromTokensToTypes(LP, Data):
    ''' Convert token-specific responsibilities into type-specific ones

    Present words will be allocated to correct topics.
    Absent words will be all assigned to top-ranked topic in that doc.

    '''
    resp = LP['resp']
    N, K = resp.shape
    if N == Data.vocab_size * Data.nDoc:
        return LP
    assert N == Data.nUniqueToken
    typeResp = np.zeros((Data.nDoc * Data.vocab_size, K))
    for d in range(Data.nDoc):
        start_d = Data.doc_range[d]
        stop_d = Data.doc_range[d+1]
        words_d = Data.word_id[start_d:stop_d]
        # Compute DocTopicCount_d for present words only
        # Purposefully ignore word_count,
        # since we want counts weighted equally by each present word
        DTC_d = np.sum(resp[start_d:stop_d, :], axis=0)
        # Get range of type-based accounting
        bstart_d = d * Data.vocab_size
        bstop_d = (d+1) * Data.vocab_size
        # Assign all background types to most common topic
        kmax = np.argmax(DTC_d)
        typeResp[bstart_d:bstop_d, kmax] = 1.0
        # Assign foreground types according to true labels
        typeResp[bstart_d + words_d, kmax] -= 1.0
        typeResp[bstart_d + words_d, :] = resp[start_d:stop_d, :]
        assert np.allclose(Data.vocab_size,
                           np.sum(typeResp[bstart_d:bstop_d]))
        assert np.allclose(resp[start_d:stop_d], typeResp[bstart_d + words_d])
    LP['resp'] = typeResp
    assert np.allclose(np.sum(typeResp, axis=1), 1.0)
    return LP

def convertLPFromTokensToDocs(LP, Data):
    ''' Convert token-specific responsibilities into document-specific ones
    '''
    resp = LP['resp']
    N, K = resp.shape
    if N == Data.nDoc:
        return LP
    docResp = np.zeros((Data.nDoc, K))
    for d in range(Data.nDoc):
        respMatForDoc = resp[Data.doc_range[d]:Data.doc_range[d + 1]]
        docResp[d, :] = np.mean(respMatForDoc, axis=0)
    LP['resp'] = docResp
    return LP


def convertLPFromDocsToTokens(LP, Data):
    ''' Convert doc-specific responsibilities into token-specific ones
    '''
    docResp = LP['resp']
    N, K = docResp.shape
    if N == Data.nUniqueToken:
        return LP
    tokResp = np.zeros((Data.nUniqueToken, K))
    for d in range(Data.nDoc):
        curDocResp = docResp[d]
        start = Data.doc_range[d]
        stop = Data.doc_range[d + 1]
        tokResp[start:stop, :] = curDocResp
    LP['resp'] = tokResp
    return LP


def convertLPFromDocsToTypes(LP, Data):
    ''' Convert doc-specific responsibilities into type-specific ones
    '''
    docResp = LP['resp']
    N, K = docResp.shape
    if N == Data.vocab_size * Data.nDoc:
        return LP
    typeResp = np.zeros((Data.vocab_size * Data.nDoc, K))
    for d in range(Data.nDoc):
        bstart = d * Data.vocab_size
        bstop = (d+1) * Data.vocab_size
        typeResp[bstart:bstop, :] = docResp[d]
    LP['resp'] = typeResp
    return LP

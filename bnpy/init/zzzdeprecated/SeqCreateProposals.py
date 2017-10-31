import numpy as np
import copy
import warnings

from bnpy.util.StateSeqUtil import calcContigBlocksFromZ
from bnpy.data.XData import XData


def proposeNewResp_randBlocks(Z_n, propResp,
                              origK=0,
                              PRNG=np.random.RandomState,
                              Kfresh=3,
                              minBlockSize=1,
                              maxBlockSize=10,
                              **kwargs):
    ''' Create new value of resp matrix with randomly-placed new blocks.

    We create Kfresh new blocks in total.
    Each one can potentially wipe out some (or all) of previous blocks.

    Returns
    -------
    propResp : 2D array of size N x Kmax
    propK : int
        total number of states used in propResp array
    '''
    # Unpack and make sure size limits work out
    T = Z_n.size
    if minBlockSize >= T:
        return propResp, origK
    maxBlockSize = np.minimum(maxBlockSize, T)

    for kfresh in range(Kfresh):
        blockSize = PRNG.randint(minBlockSize, maxBlockSize)
        a = PRNG.randint(0, T - blockSize + 1)
        b = a + blockSize
        propResp[a:b, :origK] = 0
        propResp[a:b, origK + kfresh] = 1
    return propResp, origK + Kfresh


def proposeNewResp_bisectExistingBlocks(Z_n, propResp,
                                        Data_n=None,
                                        tempModel=None,
                                        origK=0,
                                        PRNG=np.random.RandomState,
                                        Kfresh=3,
                                        PastAttemptLog=dict(),
                                        **kwargs):
    ''' Create new value of resp matrix with randomly-placed new blocks.

    We create Kfresh new blocks in total.
    Each one can potentially wipe out some (or all) of previous blocks.

    Returns
    -------
    propResp : 2D array of size N x Kmax
    propK : int
        total number of states used in propResp array
    '''
    # Iterate over current contig blocks
    blockSizes, blockStarts, blockStates = \
        calcContigBlocksFromZ(Z_n, returnStates=1)
    nBlocks = len(blockSizes)

    if 'blocks' not in PastAttemptLog:
        PastAttemptLog['blocks'] = dict()
    if 'strategy' not in PastAttemptLog:
        PastAttemptLog['strategy'] = 'byState'
        # PastAttemptLog['strategy'] = PRNG.choice(
        #    ['byState', 'bySize'])

    if PastAttemptLog['strategy'] == 'byState':
        Kcur = blockStates.max() + 1
        Kextra = Kcur - PastAttemptLog['uIDs'].size
        if Kextra > 0:
            maxUID = PastAttemptLog['maxUID']
            uIDs = PastAttemptLog['uIDs']
            for extraPos in range(Kextra):
                maxUID += 1
                uIDs = np.append(uIDs, maxUID)
            PastAttemptLog['maxUID'] = maxUID
            PastAttemptLog['uIDs'] = uIDs

        candidateStateUIDs = set()
        for state in np.unique(blockStates):
            uid = PastAttemptLog['uIDs'][state]
            candidateStateUIDs.add(uid)

        if 'nTryByStateUID' not in PastAttemptLog:
            PastAttemptLog['nTryByStateUID'] = dict()

        minTry = np.inf
        for badState, nTry in list(PastAttemptLog['nTryByStateUID'].items()):
            if badState in candidateStateUIDs:
                if nTry < minTry:
                    minTry = nTry
        untriedList = [x for x in candidateStateUIDs
                       if x not in PastAttemptLog['nTryByStateUID'] or
                       PastAttemptLog['nTryByStateUID'][x] == minTry]
        if len(untriedList) > 0:
            candidateStateUIDs = untriedList
        else:
            # Keep only candidates that have been tried the least
            for badState, nTry in list(PastAttemptLog['nTryByStateUID'].items()):
                # Remove bad State from candidateStateUIDs
                if badState in candidateStateUIDs:
                    if nTry > minTry:
                        candidateStateUIDs.remove(badState)
        candidateStateUIDs = np.asarray([x for x in candidateStateUIDs])
        # Pick a state that we haven't tried yet,
        # uniformly at random
        if len(candidateStateUIDs) > 0:
            chosenStateUID = PRNG.choice(np.asarray(candidateStateUIDs))
            chosenState = np.flatnonzero(
                chosenStateUID == PastAttemptLog['uIDs'])[0]

            chosen_mask = blockStates == chosenState
            chosenBlockIDs = np.flatnonzero(chosen_mask)

            if chosenBlockIDs.size > 1:
                # Favor blocks assigned to this state that are larger
                p = blockSizes[chosen_mask].copy()
                p /= p.sum()
                chosenBlockIDs = PRNG.choice(chosenBlockIDs,
                                             size=np.minimum(
                                                 Kfresh,
                                                 len(chosenBlockIDs)),
                                             p=p, replace=False)

            remBlockIDs = np.flatnonzero(np.logical_not(chosen_mask))
            PRNG.shuffle(remBlockIDs)
            order = np.hstack([
                chosenBlockIDs,
                remBlockIDs
            ])

        else:
            # Just use the block sizes and starts in random order
            order = PRNG.permutation(blockSizes.size)
        blockSizes = blockSizes[order]
        blockStarts = blockStarts[order]
        blockStates = blockStates[order]
    else:
        sortOrder = np.argsort(-1 * blockSizes)
        blockSizes = blockSizes[sortOrder]
        blockStarts = blockStarts[sortOrder]
        blockStates = blockStates[sortOrder]

    nBlocks = len(blockSizes)
    kfresh = 0  # number of new states added
    for blockID in range(nBlocks):
        if kfresh >= Kfresh:
            break
        a = blockStarts[blockID]
        b = blockStarts[blockID] + blockSizes[blockID]

        # Avoid overlapping with previous attempts that failed
        maxOverlapWithPreviousFailure = 0.0
        for (preva, prevb), prevm in list(PastAttemptLog['blocks'].items()):
            # skip previous attempts that succeed
            if prevm > preva:
                continue
            Tunion = np.maximum(b, prevb) - np.minimum(a, preva)
            minb = np.minimum(b, prevb)
            maxa = np.maximum(a, preva)
            if maxa < minb:
                Tintersect = minb - maxa
            else:
                Tintersect = 0
                continue
            IoU = Tintersect / float(Tunion)
            maxOverlapWithPreviousFailure = np.maximum(
                maxOverlapWithPreviousFailure, IoU)
        if maxOverlapWithPreviousFailure > 0.95:
            # print 'SKIPPING BLOCK %d,%d with overlap %.2f' % (
            #     a, b, maxOverlapWithPreviousFailure)
            continue

        stride = int(np.ceil((b - a) / 25.0))
        stride = np.maximum(1, stride)
        offset = PRNG.choice(np.arange(stride))
        a += offset
        bestm = findBestCutForBlock(Data_n, tempModel,
                                    a=a,
                                    b=b,
                                    stride=stride)

        PastAttemptLog['blocks'][(a, b)] = bestm

        print('TARGETING UID: ', PastAttemptLog['uIDs'][blockStates[blockID]])
        print('BEST BISECTION CUT: [%4d, %4d, %4d] w/ stride %d' % (
            a, bestm, b, stride))

        curUID = PastAttemptLog['uIDs'][blockStates[blockID]]
        if bestm == a:
            if curUID in PastAttemptLog['nTryByStateUID']:
                PastAttemptLog['nTryByStateUID'][curUID] += 1
            else:
                PastAttemptLog['nTryByStateUID'][curUID] = 1
        else:
            PastAttemptLog['nTryByStateUID'][curUID] = 0  # success!

        if bestm == a:
            propResp[a:b, :origK] = 0
            propResp[a:b, origK + kfresh] = 1
            kfresh += 1

        else:
            propResp[a:bestm, :origK] = 0
            propResp[a:bestm, origK + kfresh] = 1
            kfresh += 1

            if kfresh >= Kfresh:
                break

            propResp[bestm:b, :origK] = 0
            propResp[bestm:b, origK + kfresh] = 1
            kfresh += 1

    return propResp, origK + kfresh


def proposeNewResp_bisectGrownBlocks(Z_n, propResp,
                                     Data_n=None,
                                     tempModel=None,
                                     origK=0,
                                     PRNG=np.random.RandomState,
                                     Kfresh=3,
                                     growthBlockSize=10,
                                     PastAttemptLog=dict(),
                                     **kwargs):
    ''' Create new value of resp matrix with randomly-placed new blocks.

    We create Kfresh new blocks in total.
    Each one can potentially wipe out some (or all) of previous blocks.

    Returns
    -------
    propResp : 2D array of size N x Kmax
    propK : int
        total number of states used in propResp array
    '''
    # Iterate over current contig blocks
    blockSizes, blockStarts, blockStates = \
        calcContigBlocksFromZ(Z_n, returnStates=1)
    nBlocks = len(blockSizes)

    if 'blocks' not in PastAttemptLog:
        PastAttemptLog['blocks'] = dict()
    if 'strategy' not in PastAttemptLog:
        PastAttemptLog['strategy'] = 'byState'
        # PastAttemptLog['strategy'] = PRNG.choice(
        #    ['byState', 'bySize'])

    if PastAttemptLog['strategy'] == 'byState':
        Kcur = blockStates.max() + 1
        Kextra = Kcur - PastAttemptLog['uIDs'].size
        if Kextra > 0:
            maxUID = PastAttemptLog['maxUID']
            uIDs = PastAttemptLog['uIDs']
            for extraPos in range(Kextra):
                maxUID += 1
                uIDs = np.append(uIDs, maxUID)
            PastAttemptLog['maxUID'] = maxUID
            PastAttemptLog['uIDs'] = uIDs

        candidateStateUIDs = set()
        for state in np.unique(blockStates):
            uid = PastAttemptLog['uIDs'][state]
            candidateStateUIDs.add(uid)

        if 'nTryByStateUID' not in PastAttemptLog:
            PastAttemptLog['nTryByStateUID'] = dict()

        minTry = np.inf
        for badState, nTry in list(PastAttemptLog['nTryByStateUID'].items()):
            if badState in candidateStateUIDs:
                if nTry < minTry:
                    minTry = nTry
        untriedList = [x for x in candidateStateUIDs
                       if x not in PastAttemptLog['nTryByStateUID'] or
                       PastAttemptLog['nTryByStateUID'][x] == 0]
        if len(untriedList) > 0:
            candidateStateUIDs = untriedList
        else:
            # Keep only candidates that have been tried the least
            for badState, nTry in list(PastAttemptLog['nTryByStateUID'].items()):
                # Remove bad State from candidateStateUIDs
                if badState in candidateStateUIDs:
                    if nTry > minTry:
                        candidateStateUIDs.remove(badState)
        candidateStateUIDs = np.asarray([x for x in candidateStateUIDs])
        # Pick a state that we haven't tried yet,
        # uniformly at random
        if len(candidateStateUIDs) > 0:
            chosenStateUID = PRNG.choice(np.asarray(candidateStateUIDs))
            chosenState = np.flatnonzero(
                chosenStateUID == PastAttemptLog['uIDs'])[0]

            chosen_mask = blockStates == chosenState
            chosenBlockIDs = np.flatnonzero(chosen_mask)

            if chosenBlockIDs.size > 1:
                # Favor blocks assigned to this state that are larger
                p = blockSizes[chosen_mask].copy()
                p /= p.sum()
                chosenBlockIDs = PRNG.choice(chosenBlockIDs,
                                             size=np.minimum(
                                                 Kfresh,
                                                 len(chosenBlockIDs)),
                                             p=p, replace=False)

            remBlockIDs = np.flatnonzero(np.logical_not(chosen_mask))
            PRNG.shuffle(remBlockIDs)
            order = np.hstack([
                chosenBlockIDs,
                remBlockIDs
            ])

        else:
            # Just use the block sizes and starts in random order
            order = PRNG.permutation(blockSizes.size)
        blockSizes = blockSizes[order]
        blockStarts = blockStarts[order]
        blockStates = blockStates[order]
    else:
        sortOrder = np.argsort(-1 * blockSizes)
        blockSizes = blockSizes[sortOrder]
        blockStarts = blockStarts[sortOrder]
        blockStates = blockStates[sortOrder]

    nBlocks = len(blockSizes)
    kfresh = 0  # number of new states added
    for blockID in range(nBlocks):
        if kfresh >= Kfresh:
            break
        a = blockStarts[blockID]
        b = blockStarts[blockID] + blockSizes[blockID]

        # Avoid overlapping with previous attempts that failed
        maxOverlapWithPreviousFailure = 0.0
        for (preva, prevb), prevm in list(PastAttemptLog['blocks'].items()):
            # skip previous attempts that succeed
            if prevm > preva:
                continue
            Tunion = np.maximum(b, prevb) - np.minimum(a, preva)
            minb = np.minimum(b, prevb)
            maxa = np.maximum(a, preva)
            if maxa < minb:
                Tintersect = minb - maxa
            else:
                Tintersect = 0
                continue
            IoU = Tintersect / float(Tunion)
            maxOverlapWithPreviousFailure = np.maximum(
                maxOverlapWithPreviousFailure, IoU)
        if maxOverlapWithPreviousFailure > 0.95:
            continue

        stride = int(np.ceil((b - a) / 25.0))
        stride = np.maximum(1, stride)

        # If we've tried this state before and FAILED,
        # maybe its time to randomly grow this block outwards
        curUID = PastAttemptLog['uIDs'][blockStates[blockID]]
        if curUID in PastAttemptLog['nTryByStateUID']:
            nFail = PastAttemptLog['nTryByStateUID'][curUID]
            if nFail > 0:
                growthPattern = PRNG.choice(
                    ['left', 'right', 'leftandright', 'none'])
                newa = a
                newb = b
                if growthPattern.count('left'):
                    newa = a - PRNG.randint(1, growthBlockSize)
                    newa = np.maximum(newa, 0)
                if growthPattern.count('right'):
                    newb = b + PRNG.randint(1, growthBlockSize)
                    newb = np.minimum(newb, Data_n.nObs)
                a = newa
                b = newb
        bestm = findBestCutForBlock(Data_n, tempModel,
                                    a=a,
                                    b=b,
                                    stride=stride)

        PastAttemptLog['blocks'][(a, b)] = bestm

        print('TARGETING UID: ', PastAttemptLog['uIDs'][blockStates[blockID]])
        print('BEST BISECTION CUT: [%4d, %4d, %4d] w/ stride %d' % (
            a, bestm, b, stride))

        if bestm == a:
            if curUID in PastAttemptLog['nTryByStateUID']:
                PastAttemptLog['nTryByStateUID'][curUID] += 1
            else:
                PastAttemptLog['nTryByStateUID'][curUID] = 1
        else:
            PastAttemptLog['nTryByStateUID'][curUID] = 0  # success!

        if bestm == a:
            propResp[a:b, :origK] = 0
            propResp[a:b, origK + kfresh] = 1
            kfresh += 1

        else:
            propResp[a:bestm, :origK] = 0
            propResp[a:bestm, origK + kfresh] = 1
            kfresh += 1

            if kfresh >= Kfresh:
                break

            propResp[bestm:b, :origK] = 0
            propResp[bestm:b, origK + kfresh] = 1
            kfresh += 1

    return propResp, origK + kfresh


def proposeNewResp_subdivideExistingBlocks(Z_n, propResp,
                                           origK=0,
                                           PRNG=np.random.RandomState,
                                           nStatesToEdit=3,
                                           Kfresh=5,
                                           minBlockSize=1,
                                           maxBlockSize=10,
                                           **kwargs):
    ''' Create new value of resp matrix with new blocks.

    We select nStatesToEdit states to change.
    For each one, we take each contiguous block,
        defined by interval [a,b]
        and subdivide that interval into arbitrary number of states
            [a, l1, l2, l3, ... lK, b]
        where the length of each new block is drawn from
            l_i ~ uniform(minBlockSize, maxBlockSize)

    Returns
    -------
    propResp : 2D array of size N x Kmax
    propK : int
        total number of states used in propResp array
    '''
    # Unpack and make sure size limits work out
    T = Z_n.size
    if minBlockSize >= T:
        return propResp, origK
    maxBlockSize = np.minimum(maxBlockSize, T)

    blockSizes, blockStarts = calcContigBlocksFromZ(Z_n)
    nBlocks = len(blockSizes)

    candidateStateIDs = list()
    candidateBlockIDsByState = dict()
    for blockID in range(nBlocks):
        stateID = Z_n[blockStarts[blockID]]
        if blockSizes[blockID] >= minBlockSize:
            candidateStateIDs.append(stateID)
            if stateID not in candidateBlockIDsByState:
                candidateBlockIDsByState[stateID] = list()
            candidateBlockIDsByState[stateID].append(blockID)

    if len(candidateStateIDs) == 0:
        return propResp, origK
    selectedStateIDs = PRNG.choice(candidateStateIDs,
                                   size=np.minimum(
                                       len(candidateStateIDs),
                                       nStatesToEdit),
                                   replace=False)

    kfresh = origK
    for stateID in selectedStateIDs:
        if kfresh >= Kfresh:
            break

        # Find contig blocks assigned to this state
        for blockID in candidateBlockIDsByState[stateID]:
            if kfresh >= Kfresh:
                break
            a = blockStarts[blockID]
            b = a + blockSizes[blockID]
            maxSize = np.minimum(b - a, maxBlockSize)
            avgSize = (maxSize + minBlockSize) / 2
            expectedLen = avgSize * Kfresh
            if expectedLen < (b - a):
                intervalLocs = [PRNG.randint(a, b - expectedLen)]
            else:
                intervalLocs = [a]
            for ii in range(Kfresh):
                nextBlockSize = PRNG.randint(minBlockSize, maxSize)
                intervalLocs.append(nextBlockSize + intervalLocs[ii])
                if intervalLocs[ii + 1] >= b:
                    break
            intervalLocs = np.asarray(intervalLocs, dtype=np.int32)
            intervalLocs = np.minimum(intervalLocs, b)
            # print 'Current interval   : [ %d, %d]' % (a, b)
            # print 'Subdivided interval: ', intervalLocs
            for iID in range(intervalLocs.size - 1):
                if kfresh >= Kfresh:
                    break
                prevLoc = intervalLocs[iID]
                curLoc = intervalLocs[iID + 1]
                propResp[prevLoc:curLoc, :] = 0
                propResp[prevLoc:curLoc, kfresh] = 1
                kfresh += 1
    assert kfresh >= origK
    return propResp, kfresh


def proposeNewResp_uniquifyExistingBlocks(Z_n, propResp,
                                          tempSS=None,
                                          origK=0,
                                          PRNG=np.random.RandomState,
                                          nStatesToEdit=None,
                                          Kfresh=5,
                                          minBlockSize=1,
                                          maxBlockSize=10,
                                          **kwargs):
    ''' Create new resp matrix with new unique blocks from existing blocks.

    We select at most nStatesToEdit states to change,
    where each one has multiple contiguous blocks.

    For each one, we take all its contiguous blocks,
    defined by intervals [a1,b1], [a2,b2], ... [aN, bN], ...
    and make a unique state for each interval.

    Returns
    -------
    propResp : 2D array of size N x Kmax
    propK : int
        total number of states used in propResp array
    '''
    if nStatesToEdit is None:
        nStatesToEdit = Kfresh

    # Unpack and make sure size limits work out
    T = Z_n.size
    if minBlockSize >= T:
        return propResp, origK
    maxBlockSize = np.minimum(maxBlockSize, T)

    blockSizes, blockStarts = calcContigBlocksFromZ(Z_n)
    nBlocks = len(blockSizes)

    candidateBlockIDsByState = dict()
    for blockID in range(nBlocks):
        stateID = Z_n[blockStarts[blockID]]
        if stateID not in candidateBlockIDsByState:
            candidateBlockIDsByState[stateID] = list()
        candidateBlockIDsByState[stateID].append(blockID)

    candidateStateIDs = list()
    for stateID in list(candidateBlockIDsByState.keys()):
        hasJustOneBlock = len(candidateBlockIDsByState[stateID]) < 2
        if tempSS is None:
            appearsOnlyInThisSeq = True
        else:
            appearsOnlyInThisSeq = tempSS.N[stateID] < 1.0

        if hasJustOneBlock and appearsOnlyInThisSeq:
            del candidateBlockIDsByState[stateID]
        else:
            candidateStateIDs.append(stateID)

    if len(candidateStateIDs) == 0:
        return propResp, origK
    selectedStateIDs = PRNG.choice(candidateStateIDs,
                                   size=np.minimum(
                                       len(candidateStateIDs),
                                       nStatesToEdit),
                                   replace=False)

    kfresh = 0
    for stateID in selectedStateIDs:
        if kfresh >= Kfresh:
            break
        # Make each block assigned to this state its own unique proposed state
        for blockID in candidateBlockIDsByState[stateID]:
            if kfresh >= Kfresh:
                break
            a = blockStarts[blockID]
            b = a + blockSizes[blockID]
            propResp[a:b, :] = 0
            propResp[a:b, origK + kfresh] = 1
            kfresh += 1
    return propResp, origK + kfresh


def proposeNewResp_dpmixture(Z_n, propResp,
                             tempModel=None,
                             tempSS=None,
                             Data_n=None,
                             origK=0,
                             Kfresh=3,
                             nVBIters=3,
                             PRNG=None,
                             PastAttemptLog=dict(),
                             **kwargs):
    ''' Create new resp matrix by DP mixture clustering of subsampled data.

    Returns
    -------
    propResp : 2D array, N x K'
    '''
    # Avoid circular imports
    from bnpy.allocmodel import DPMixtureModel
    from bnpy import HModel
    from bnpy.mergemove import MergePlanner, MergeMove

    # Select ktarget
    if 'strategy' not in PastAttemptLog:
        PastAttemptLog['strategy'] = 'byState'

    if PastAttemptLog['strategy'] == 'byState':
        Kcur = tempModel.obsModel.K
        Kextra = Kcur - PastAttemptLog['uIDs'].size
        if Kextra > 0:
            maxUID = PastAttemptLog['maxUID']
            uIDs = PastAttemptLog['uIDs']
            for extraPos in range(Kextra):
                maxUID += 1
                uIDs = np.append(uIDs, maxUID)
            PastAttemptLog['maxUID'] = maxUID
            PastAttemptLog['uIDs'] = uIDs

        candidateStateUIDs = set()
        for state in np.unique(Z_n):
            uid = PastAttemptLog['uIDs'][state]
            candidateStateUIDs.add(uid)
        allAvailableUIDs = [x for x in candidateStateUIDs]
        if 'nTryByStateUID' not in PastAttemptLog:
            PastAttemptLog['nTryByStateUID'] = dict()

        minTry = np.inf
        for badState, nTry in list(PastAttemptLog['nTryByStateUID'].items()):
            if badState in candidateStateUIDs:
                if nTry < minTry:
                    minTry = nTry
        untriedList = [x for x in candidateStateUIDs
                       if x not in PastAttemptLog['nTryByStateUID'] or
                       PastAttemptLog['nTryByStateUID'][x] == 0]
        if len(untriedList) > 0:
            candidateStateUIDs = untriedList
        else:
            # Keep only candidates that have been tried the least
            for badState, nTry in list(PastAttemptLog['nTryByStateUID'].items()):
                # Remove bad State from candidateStateUIDs
                if badState in candidateStateUIDs:
                    if nTry > minTry:
                        candidateStateUIDs.remove(badState)
        candidateStateUIDs = np.asarray([x for x in candidateStateUIDs])
        # Pick a state that we haven't tried yet,
        # uniformly at random
        if len(candidateStateUIDs) > 0:
            chosenStateUID = PRNG.choice(np.asarray(candidateStateUIDs))
            ktarget = np.flatnonzero(
                chosenStateUID == PastAttemptLog['uIDs'])[0]
        else:
            # Just pick a target at random
            chosenStateUID = PRNG.choice(np.asarray(allAvailableUIDs))

    ktarget = np.flatnonzero(
        chosenStateUID == PastAttemptLog['uIDs'])[0]

    relDataIDs = np.flatnonzero(Z_n == ktarget)

    # If the selected state is too small, just make a new state for all relIDs
    if relDataIDs.size < Kfresh:
        if chosenStateUID in PastAttemptLog['nTryByStateUID']:
            PastAttemptLog['nTryByStateUID'][chosenStateUID] += 1
        else:
            PastAttemptLog['nTryByStateUID'][chosenStateUID] = 1
        propResp[relDataIDs, :] = 0
        propResp[relDataIDs, origK + 1] = 1
        return propResp, origK + 1

    if hasattr(Data_n, 'Xprev'):
        Xprev = Data_n.Xprev[relDataIDs]
    else:
        Xprev = None
    targetData = XData(X=Data_n.X[relDataIDs],
                       Xprev=Xprev)

    myDPModel = DPMixtureModel('VB', gamma0=10)
    myObsModel = copy.deepcopy(tempModel.obsModel)
    delattr(myObsModel, 'Post')
    myObsModel.ClearCache()

    myHModel = HModel(myDPModel, myObsModel)
    initname = PRNG.choice(['randexamplesbydist', 'randcontigblocks'])
    myHModel.init_global_params(targetData, K=Kfresh,
                                initname=initname,
                                **kwargs)

    Kfresh = myHModel.obsModel.K
    mergeIsPromising = True
    while Kfresh > 1 and mergeIsPromising:
        for vbiter in range(nVBIters):
            targetLP = myHModel.calc_local_params(targetData)
            targetSS = myHModel.get_global_suff_stats(targetData, targetLP)
            # Delete unnecessarily small comps
            if vbiter == nVBIters - 1:
                smallIDs = np.flatnonzero(targetSS.getCountVec() <= 1)
                for kdel in reversed(smallIDs):
                    if targetSS.K > 1:
                        targetSS.removeComp(kdel)
            # Global step
            myHModel.update_global_params(targetSS)

        # Do merges
        mPairIDs, MM = MergePlanner.preselect_candidate_pairs(
            myHModel, targetSS,
            preselect_routine='wholeELBO',
            doLimitNumPairs=0,
            returnScoreMatrix=1,
            **kwargs)
        targetLP = myHModel.calc_local_params(
            targetData, mPairIDs=mPairIDs, limitMemoryLP=1)
        targetSS = myHModel.get_global_suff_stats(
            targetData, targetLP,
            mPairIDs=mPairIDs,
            doPrecompEntropy=1,
            doPrecompMergeEntropy=1)
        myHModel.update_global_params(targetSS)
        curELBO = myHModel.calc_evidence(SS=targetSS)
        myHModel, targetSS, curELBO, Info = MergeMove.run_many_merge_moves(
            myHModel, targetSS, curELBO,
            mPairIDs, M=MM,
            isBirthCleanup=1)
        mergeIsPromising = len(Info['AcceptedPairs']) > 0
        Kfresh = targetSS.K

    if mergeIsPromising:
        targetLP = myHModel.calc_local_params(targetData)
    propResp[relDataIDs, :] = 0
    propResp[relDataIDs, origK:origK + Kfresh] = targetLP['resp']

    # Test if we added at least 2 states with mass > 1
    didAddNonEmptyNewStates = np.sum(targetSS.N > 1.0) >= 2
    print('dpmixture proposal: targetUID %d didAddNonEmptyNewStates %d' % (
        chosenStateUID, didAddNonEmptyNewStates))
    if didAddNonEmptyNewStates:
        print('NEW STATE MASSES:', end=' ')
        print(' '.join(['%5.1f' % (x) for x in targetSS.N]))
        PastAttemptLog['nTryByStateUID'][chosenStateUID] = 0  # success!
    else:
        if chosenStateUID in PastAttemptLog['nTryByStateUID']:
            PastAttemptLog['nTryByStateUID'][chosenStateUID] += 1
        else:
            PastAttemptLog['nTryByStateUID'][chosenStateUID] = 1
    return propResp, origK + Kfresh


def findBestCutForBlock(Data_n, tempModel,
                        a=0, b=400,
                        stride=3):
    ''' Search for best cut point over interval [a,b] in provided sequence n.
    '''
    tempModel = tempModel.copy()

    def calcObsModelELBOForInterval(SSab):
        tempModel.obsModel.update_global_params(SSab)
        ELBOab = tempModel.obsModel.calc_evidence(None, SSab, None)
        return ELBOab

    SSab = tempModel.obsModel.calcSummaryStatsForContigBlock(
        Data_n, a=a, b=b)
    ELBOab = calcObsModelELBOForInterval(SSab)

    # Initialize specific suff stat bags for intervals [a,m] and [m,b]
    SSmb = SSab
    SSam = SSab.copy()
    SSam.setAllFieldsToZero()
    assert np.allclose(SSam.N.sum() + SSmb.N.sum(), b - a)

    score = -1 * np.inf * np.ones(b - a)
    score[0] = ELBOab
    for m in np.arange(a + stride, b, stride):
        assert m > a
        assert m < b
        # Grab segment recently converted to [a,m] interval
        SSstride = tempModel.obsModel.calcSummaryStatsForContigBlock(
            Data_n, a=(m - stride), b=m)
        SSam += SSstride
        SSmb -= SSstride
        assert np.allclose(SSam.N.sum() + SSmb.N.sum(), b - a)

        ELBOam = calcObsModelELBOForInterval(SSam)
        ELBOmb = calcObsModelELBOForInterval(SSmb)
        score[m - a] = ELBOam + ELBOmb
        # print a, m, b, 'score %.3e  Nam %.3f' % (score[m - a], SSam.N[0])

    bestm = a + np.argmax(score)
    return bestm

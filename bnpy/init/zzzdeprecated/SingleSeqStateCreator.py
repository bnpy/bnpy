import copy
import numpy as np
import warnings

from bnpy.deletemove.DEvaluator import runDeleteMoveAndUpdateMemory
from bnpy.util.StateSeqUtil import calcContigBlocksFromZ

from bnpy.allocmodel.hmm.HDPHMMUtil import calcELBOForSingleSeq_FromLP

from bnpy.init.SeqCreatePlanner import calcEvalMetricForStates_KL
from bnpy.init.SeqCreateRefinery import \
    refineProposedRespViaLocalGlobalStepsAndDeletes

# All proposal generating functions, of the form
#   proposeNewResp_<creationProposalName>
from bnpy.init.SeqCreateProposals import *


def createSingleSeqLPWithNewStates_ManyProposals(Data_n, LP_n, model,
                                                 SS=None,
                                                 lapFrac=0,
                                                 PastAttemptLog=None,
                                                 nDocSeenForProposal=0,
                                                 nDocSeenInCurLap=0,
                                                 **kwargs):
    ''' Perform many proposal moves on provided sequence.

    Args
    ----
    SS : SuffStatBag
        represents whole dataset seen thus far,
        including *previous* stats from current sequence n

    Returns
    -------
    LP_n
    tempModel : HModel
        represents whole dataset seen thus far
        with all global params fully updated from tempSS suff stats
    tempSS : SuffStatBag
        represents whole dataset seen thus far
        including TWO versions of stats for current sequence n
    '''
    if 'verbose' not in kwargs:
        kwargs['verbose'] = 0

    if lapFrac > 1:
        # assert SS.nDoc == Data_n.nDocTotal + nDocSeenForProposal
        assert SS.nDoc >= Data_n.nDocTotal
    else:
        if SS is not None:
            assert SS.nDoc == nDocSeenInCurLap + nDocSeenForProposal

    if SS is not None:
        tempSS = SS.copy(includeELBOTerms=False, includeMergeTerms=False)
    else:
        tempSS = None

    if PastAttemptLog is None:
        PastAttemptLog = dict()

    creationProposalNames = kwargs['creationProposalName'].split('+')
    creationNumProposal = kwargs['creationNumProposal']

    if creationNumProposal < 1.0:
        beyondReqFrac = (
            lapFrac - np.floor(lapFrac)) > creationNumProposal + 1e-5
        isLastLap = np.allclose(lapFrac, np.ceil(lapFrac))
        if beyondReqFrac or isLastLap:
            creationNumProposal = 0
        else:
            creationNumProposal = 1
    else:
        creationNumProposal = int(creationNumProposal)

    origK = model.obsModel.K
    propID = 0
    for creationProposalName in creationProposalNames:
        for repID in range(creationNumProposal):
            if propID > 0:
                # Remove current stats for this seq before next proposal.
                SS_n = model.get_global_suff_stats(Data_n, LP_n)
                tempSS -= SS_n

            if kwargs['verbose']:
                print('======== lapFrac %.3f seqName %s | proposal %d/%d' % (
                    lapFrac, kwargs['seqName'],
                    propID + 1, creationNumProposal))
                if propID == 0:
                    print('   BEFORE tempSS.nDoc %d' % (tempSS.nDoc))
                    print('N ', ' '.join(
                        ['%5.1f' % (x) for x in tempSS.N[:20]]))

            if lapFrac > 1:
                assert tempSS.nDoc == Data_n.nDocTotal + nDocSeenForProposal
            else:
                if tempSS is not None:
                    assert tempSS.nDoc == nDocSeenInCurLap + \
                        nDocSeenForProposal

            curKwargs = dict(**kwargs)
            curKwargs['creationProposalName'] = creationProposalName
            LP_n, model, tempSS = createSingleSeqLPWithNewStates(
                Data_n, LP_n, model,
                SS=tempSS,
                PastAttemptLog=PastAttemptLog,
                **curKwargs)
            propID += 1

            if kwargs['verbose']:
                print('--- AFTER tempSS.nDoc %d' % (tempSS.nDoc))
                print('N ', ' '.join(
                    ['%5.1f' % (x) for x in tempSS.N[:20]]))

            if lapFrac > 1:
                assert tempSS.nDoc == nDocSeenForProposal + 1 + \
                    Data_n.nDocTotal
            else:
                assert tempSS.nDoc == nDocSeenForProposal + 1 + \
                    nDocSeenInCurLap

    T_n = LP_n['resp'].shape[0]
    assert tempSS.N.sum() >= T_n - 1e-5

    didAnyProposals = propID > 0
    if didAnyProposals:
        if lapFrac > 1:
            # Will have two versions of the current sequence
            Norig = SS.N.sum()
            assert tempSS.N.sum() >= Norig + T_n - 1e-5
            # assert tempSS.nDoc == Data_n.nDocTotal + nDocSeenForProposal + 1
        # else:
        #    assert tempSS.nDoc == nDocSeenForProposal + 1 + \
        #        nDocSeenInCurLap

    # else:
    #    if lapFrac > 1:
    #        assert tempSS.nDoc == Data_n.nDocTotal
    #    else:
    #        assert tempSS.nDoc == nDocSeenForProposal + 1 + \
    #            nDocSeenInCurLap

    Info = dict(
        didAnyProposals=didAnyProposals,
        origK=origK,
        propK=tempSS.K,
    )

    if 'blocks' in PastAttemptLog:
        del PastAttemptLog['blocks']
    return LP_n, model, tempSS, Info


def createSingleSeqLPWithNewStates(Data_n, LP_n, hmodel,
                                   SS=None,
                                   verbose=0,
                                   Kfresh=3,
                                   minBlockSize=20,
                                   maxBlockSize=np.inf,
                                   creationProposalName='mixture',
                                   PRNG=None,
                                   seed=0,
                                   n=0,
                                   Kmax=200,
                                   seqName='',
                                   **kwargs):
    ''' Try a proposal for new LP for current sequence n.

    Args
    -------
    SS : SuffStatBag
        represents whole-dataset seen thus far
        if first lap:
            does NOT include stats from n
        else:
            includes *previous* stats from current sequence n

    Returns
    -------
    LP_n : dict of local params, with K + Knew states (Knew >= 0)
    tempModel : HModel
        represents whole dataset seen thus far
        with all global params fully updated from tempSS suff stats
    tempSS : SuffStatBag
        represents whole dataset seen thus far
        if first lap:
            include only the most recent stats from n
            whether accepted or rejected.
        else
            includes *previous* stats from current sequence n
            and the most recent stats from n
    '''
    kwargs['creationProposalName'] = creationProposalName
    kwargs['minBlockSize'] = minBlockSize
    kwargs['maxBlockSize'] = maxBlockSize
    kwargs['PRNG'] = PRNG

    resp_n = LP_n['resp']
    N, origK = resp_n.shape
    if origK >= Kmax:
        if kwargs['doVizSeqCreate']:
            print('skipped. at max capacity for states.')
        SS_n = hmodel.get_global_suff_stats(Data_n, LP_n)
        if SS is None:
            SS = SS_n.copy()
        else:
            SS += SS_n
        assert SS.N.sum() >= LP_n['resp'].shape[0] - 1e-5
        return LP_n, hmodel, SS
    else:
        Kfresh = np.minimum(Kfresh, Kmax - origK)
    # Set Kfresh AFTER it has been adjusted to avoid going over capacity
    kwargs['Kfresh'] = Kfresh

    if PRNG is None:
        PRNG = np.random.RandomState(seed)
    tempModel = hmodel.copy()
    tempSS = SS
    if tempSS is not None:
        tempSS = tempSS.copy()
        if hasattr(tempSS, 'uIDs'):
            delattr(tempSS, 'uIDs')

    propResp = np.zeros((N, origK + Kfresh))
    propResp[:, :origK] = resp_n
    Z_n = np.argmax(resp_n, axis=1)

    # Select the state to target in this sequence.
    # Must appear with sufficient size.
    uIDs = np.unique(Z_n)
    sizes = np.asarray([np.sum(Z_n == uID) for uID in uIDs])
    elig_mask = sizes >= minBlockSize
    if np.sum(elig_mask) > 0:
        sizes = sizes[elig_mask]
        uIDs = uIDs[elig_mask]
        ktarget = PRNG.choice(uIDs)
    else:
        ktarget = PRNG.choice(uIDs)

    # Prepare all the arguments to pass to proposal function
    allPropKwArgs = dict(
        tempModel=tempModel,
        tempSS=tempSS,
        Data_n=Data_n,
        ktarget=ktarget,
        PRNG=PRNG,
        origK=origK,
        Kfresh=Kfresh,
        minBlockSize=minBlockSize,
        maxBlockSize=maxBlockSize,
    )
    allPropKwArgs.update(kwargs)

    # Execute the proposal. Calls a function imported from SeqCreateProposals.
    propFuncName = 'proposeNewResp_' + creationProposalName
    GlobalVars = globals()
    if propFuncName in GlobalVars:
        propFunc = GlobalVars[propFuncName]
        propResp, propK = propFunc(Z_n, propResp, **allPropKwArgs)
    else:
        msg = "Unrecognized creationProposalName: %s" % (creationProposalName)
        raise NotImplementedError(msg)

    if propK == origK and np.allclose(Z_n, propResp.argmax(axis=1)):
        if kwargs['doVizSeqCreate']:
            print('rejected. proposal is no change from original.')
        SS_n = hmodel.get_global_suff_stats(Data_n, LP_n)
        if SS is None:
            SS = SS_n.copy()
        else:
            SS += SS_n
        assert SS.N.sum() >= Z_n.size - 1e-5
        return LP_n, hmodel, SS

    # Create complete LP dict from the proposed segmentation
    propLP_n = tempModel.allocModel.initLPFromResp(
        Data_n, dict(resp=propResp[:, :propK]))

    # Refine this set of local parameters
    propLP_n, tempModel, tempSS, Info = \
        refineProposedRespViaLocalGlobalStepsAndDeletes(
            Data_n, propLP_n, tempModel, tempSS,
            propK=propK,
            origK=origK,
            verbose=verbose,
            **kwargs)

    if origK == tempSS.K and np.allclose(Z_n, propLP_n['resp'].argmax(axis=1)):
        if kwargs['doVizSeqCreate']:
            print('rejected. proposal is no change from original.')
        SS_n = hmodel.get_global_suff_stats(Data_n, LP_n)
        if SS is None:
            SS = SS_n.copy()
        else:
            SS += SS_n
        assert SS.N.sum() >= Z_n.size - 1e-5
        return LP_n, hmodel, SS

    # EDIT: the single sequence ELBO is really not a good metric.
    # It can downweight a proposal that takes a crappy state used in many seq
    # and replaces it with a new cluster for this sequence only,
    # because it only looks at the current sequence's data.
    # propScore = calcELBOForSingleSeq_FromLP(Data_n, propLP_n, hmodel)
    # curScore = calcELBOForSingleSeq_FromLP(Data_n, LP_n, hmodel)
    propScore = propLP_n['evidence']
    curScore = LP_n['evidence']
    doAccept = propScore > curScore + 1e-6

    if kwargs['doVizSeqCreate']:
        print(seqName)
        print('propLP K %3d evidence %.3f  score %.6f' % (
            propLP_n['resp'].shape[1], propLP_n['evidence'], propScore))
        print(' curLP K %3d evidence %.3f  score %.6f' % (
            LP_n['resp'].shape[1], LP_n['evidence'], curScore))

        if doAccept:
            Nnew = propLP_n['resp'][:, origK:].sum(axis=0)
            massNew_str = ' '.join(['%.2f' % (x) for x in Nnew])
            print('ACCEPTED! Mass of new states: [%s]' % (massNew_str))
        else:
            print('rejected')

        kwargs.update(Info)

        if kwargs['doVizSeqCreate'] > 1:
            doShow = doAccept
        else:
            doShow = True
        if doShow:
            showProposal(Data_n, LP_n, propResp, propLP_n,
                         doAccept=doAccept,
                         n=n,
                         seqName=seqName,
                         curModel=hmodel,
                         **kwargs)
            print('')
            print('')
            print('')

    if doAccept:
        assert tempSS.N.sum() >= Z_n.size - 1e-5
        return propLP_n, tempModel, tempSS
    else:
        SS_n = hmodel.get_global_suff_stats(Data_n, LP_n)
        if SS is None:
            SS = SS_n.copy()
        else:
            SS += SS_n
        assert SS.N.sum() >= Z_n.size - 1e-5
        return LP_n, hmodel, SS


def showProposal(Data_n, LP_n, propResp, propLP_n,
                 n=0,
                 seqName='',
                 extraIDs_remaining=None,
                 doAccept=None,
                 doCompactTrueLabels=False,
                 Kfresh=None,
                 creationProposalName=None,
                 curModel=None,
                 **kwargs):
    from matplotlib import pylab
    from bnpy.util.StateSeqUtil import alignEstimatedStateSeqToTruth
    from bnpy.util.StateSeqUtil import makeStateColorMap

    Z_n = LP_n['resp'].argmax(axis=1)
    if hasattr(Data_n, 'TrueParams') and 'Z' in Data_n.TrueParams:
        Ztrue = np.asarray(Data_n.TrueParams['Z'], np.int32)
    else:
        Ztrue = Z_n.copy()

    if doCompactTrueLabels:
        # Map Ztrue to compact set of uniqueIDs
        # that densely covers 0, 1, 2, ... Kunique
        # instead of (possibly) covering 0, 1, 2, ... Ktrue, Ktrue >> Kunique
        uLabels = np.unique(Ztrue)
        ZtrueA = -1 * np.ones_like(Ztrue)
        for uLoc, uID in enumerate(uLabels):
            mask = Ztrue == uID
            ZtrueA[mask] = uLoc
        assert np.all(ZtrueA >= 0)
        assert np.all(ZtrueA < len(uLabels))
        Ztrue = ZtrueA

    curZA, AlignInfo = alignEstimatedStateSeqToTruth(
        Z_n, Ztrue, returnInfo=1)

    Kcur = LP_n['resp'].shape[1]
    nTrue = Ztrue.max() + 1
    nExtra = curZA.max() + 1 - nTrue
    Kmax = np.maximum(nTrue, nTrue + nExtra)

    propZstart = propResp.argmax(axis=1)
    propZAstart = alignEstimatedStateSeqToTruth(
        propZstart, Ztrue, useInfo=AlignInfo)

    # Relabel each unique state represented in propZrefined
    # so that state ids correspond to those in propZstart

    # Step 1: propZrefined states all have negative ids
    Kprop = propLP_n['resp'].shape[1]
    propZrefined = -1 * propLP_n['resp'].argmax(axis=1) - 1
    for origLoc, kk in enumerate(range(Kcur, Kprop)):
        propZrefined[propZrefined == -1 * kk - 1] = extraIDs_remaining[origLoc]
    # Transform any original states back to original ids
    propZrefined[propZrefined < 0] = -1 * propZrefined[propZrefined < 0] - 1
    # Align to true states
    propZArefined = alignEstimatedStateSeqToTruth(
        propZrefined, Ztrue, useInfo=AlignInfo)

    nHighlight = propZAstart.max() + 1 - Kmax

    cmap = makeStateColorMap(nTrue=nTrue,
                             nExtra=np.maximum(0, nExtra),
                             nHighlight=np.maximum(0, nHighlight))

    Kmaxxx = np.maximum(Kmax, propZAstart.max() + 1)
    Kmaxxx = np.maximum(Kmaxxx, propZArefined.max() + 1)
    imshowArgs = dict(interpolation='nearest',
                      aspect=Z_n.size / 1.0,
                      cmap=cmap,
                      vmin=0, vmax=Kmaxxx - 1)
    pylab.subplots(nrows=4, ncols=1, figsize=(12, 5))
    # show ground truth
    # print 'UNIQUE IDS in each aligned Z'
    # print np.unique(Ztrue)
    # print np.unique(curZA)
    # print np.unique(propZstart), '>', np.unique(propZAstart)
    # print np.unique(propZrefined), '>', np.unique(propZArefined)

    ax = pylab.subplot(4, 1, 1)
    pylab.imshow(Ztrue[np.newaxis, :], **imshowArgs)
    pylab.title('Ground truth' + ' ' + seqName)
    pylab.yticks([])
    # show current
    pylab.subplot(4, 1, 2, sharex=ax)
    pylab.imshow(curZA[np.newaxis, :], **imshowArgs)
    pylab.title('Current estimate')
    pylab.yticks([])
    # show init
    pylab.subplot(4, 1, 3, sharex=ax)
    pylab.imshow(propZAstart[np.newaxis, :], **imshowArgs)
    pylab.title('Initial proposal')
    pylab.yticks([])

    # show final
    pylab.subplot(4, 1, 4, sharex=ax)
    pylab.imshow(propZArefined[np.newaxis, :], **imshowArgs)
    if doAccept is None:
        acceptMsg = ''
    elif doAccept:
        acceptMsg = '  ACCEPTED!'
    else:
        acceptMsg = '  REJECTED!'

    pylab.title('Refined proposal' + acceptMsg)
    pylab.yticks([])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        pylab.tight_layout()
    pylab.show(block=False)
    keypress = input("Press any key to continue>>>")
    if keypress.count('embed'):
        from IPython import embed
        embed()
    pylab.close()


def initSingleSeq_SeqAllocContigBlocks(n, Data, hmodel,
                                       SS=None,
                                       Kmax=50,
                                       initBlockLen=20,
                                       verbose=0,
                                       **kwargs):
    ''' Initialize single sequence using new states and existing ones.

    Returns
    -------
    hmodel : HModel
        represents whole dataset
    SS : SuffStatBag
        represents whole dataset
    '''
    if verbose:
        print('<<<<<<<<<<<<<<<< Creating states for seq. %d' % (n))

    assert hasattr(Data, 'nDoc')
    Data_n = Data.select_subset_by_mask([n], doTrackTruth=1)
    obsModel = hmodel.obsModel

    resp_n = np.zeros((Data_n.nObs, Kmax))
    if SS is None:
        K = 1
        SSobsonly = None
    else:
        K = SS.K + 1
        SSobsonly = SS.copy()

    SSprevComp = None

    blockTuples = getListOfContigBlocks(Data_n)
    for blockID, (a, b) in enumerate(blockTuples):
        SSab = obsModel.calcSummaryStatsForContigBlock(
            Data_n, a=a, b=b)

        if hasattr(Data_n, 'TrueParams'):
            Zab = Data_n.TrueParams['Z'][a:b]
            trueStateIDstr = ', '.join(['%d:%d' % (kk, np.sum(Zab == kk))
                                        for kk in np.unique(Zab)])
            trueStateIDstr = ' truth: ' + trueStateIDstr
        else:
            trueStateIDstr = ''

        if blockID == 0:
            SSprevComp = SSab
            resp_n[a:b, K - 1] = 1
            if verbose >= 2:
                print("block %d/%d: %d-%d" % (blockID, len(blockTuples), a, b), end=' ')
                print('assigned to first state %d' % (K - 1), trueStateIDstr)
            continue

        # Should we merge current interval [a,b] with previous state?
        ELBOimprovement = obsModel.calcHardMergeGap_SpecificPairSS(
            SSprevComp, SSab)
        if (ELBOimprovement >= -0.000001):
            # Positive means we merge block [a,b] with previous state
            SSprevComp += SSab
            resp_n[a:b, K - 1] = 1
            if verbose >= 2:
                print("block %d/%d: %d-%d" % (blockID, len(blockTuples), a, b), end=' ')
                print('building on existing state %d' % (K - 1), trueStateIDstr)

        else:
            # Insert finished block as a new component
            if SSobsonly is None:
                SSobsonly = SSprevComp
            else:
                # Remove any keys associated with alloc model
                for key in list(SSobsonly._FieldDims.keys()):
                    if key not in SSprevComp._FieldDims:
                        del SSobsonly._FieldDims[key]
                SSobsonly.insertComps(SSprevComp)

            # Try to merge this new state into an already existing state
            resp_n, K, SSobsonly = mergeDownLastStateIfPossible(
                resp_n, K, SSobsonly, SSprevComp, obsModel, verbose=verbose)

            # Assign block [a,b] to a new state!
            K += 1
            resp_n[a:b, K - 1] = 1
            SSprevComp = SSab
            if verbose >= 2:
                print("block %d/%d: %d-%d" % (blockID, len(blockTuples), a, b), end=' ')
                print('building on existing state %d' % (K - 1), trueStateIDstr)

    # Deal with final block
    if SSobsonly is not None:
        # Remove any keys associated with alloc model
        for key in list(SSobsonly._FieldDims.keys()):
            if key not in SSprevComp._FieldDims:
                del SSobsonly._FieldDims[key]
        SSobsonly.insertComps(SSprevComp)
        resp_n, K, SSobsonly = mergeDownLastStateIfPossible(
            resp_n, K, SSobsonly, SSprevComp, obsModel, verbose=verbose)
    del SSobsonly

    hmodel, SS, SS_n = refineSegmentationViaLocalGlobalSteps(
        SS, hmodel, Data_n, resp_n, K,
        verbose=verbose)

    hmodel, SS = removeSmallUniqueCompsViaDelete(
        SS, hmodel, Data_n, SS_n,
        verbose=verbose,
        initBlockLen=initBlockLen)

    return hmodel, SS


def refineSegmentationViaLocalGlobalSteps(
        SS, hmodel, Data_n, resp_n, K,
        verbose=0,
        nSteps=3,
):
    '''

    Returns
    -------
    hmodel : HModel
        consistent with entire dataset
    SS : SuffStatBag
        representing whole dataset
    SS_n : SuffStatBag
        represents only current sequence n
    '''
    for rep in range(nSteps):
        if rep == 0:
            LP_n = dict(resp=resp_n[:, :K])
            LP_n = hmodel.allocModel.initLPFromResp(Data_n, LP_n)
        else:
            LP_n = hmodel.calc_local_params(Data_n)

        SS_n = hmodel.get_global_suff_stats(Data_n, LP_n)
        assert np.allclose(SS_n.N.sum(), Data_n.nObs)

        # Update whole-dataset stats
        if rep == 0:
            prevSS_n = SS_n
            if SS is None:
                SS = SS_n.copy()
            else:
                SS.insertEmptyComps(SS_n.K - SS.K)
                SS += SS_n
        else:
            SS -= prevSS_n
            SS += SS_n
            prevSS_n = SS_n

        # Reorder the states from big to small
        # using aggregate stats from entire dataset
        if rep == nSteps - 1:
            order = np.argsort(-1 * SS.N)
            SS_n.reorderComps(order)
            SS.reorderComps(order)
        hmodel.update_global_params(SS)
        for i in range(3):
            hmodel.allocModel.update_global_params(SS)
    return hmodel, SS, SS_n


def removeSmallUniqueCompsViaDelete(
        SS, hmodel, Data_n, SS_n,
        initBlockLen=20,
        verbose=0):
    ''' Remove small comps unique to sequence n if accepted by delete move.

    Returns
    -------
    hmodel : HModel
        consistent with entire dataset
    SS : SuffStatBag
        representing whole dataset
    '''
    # Try deleting any UIDs unique to this sequence with small size
    K_n = np.sum(SS_n.N > 0.01)
    IDs_n = np.flatnonzero(SS_n.N <= initBlockLen)
    candidateUIDs = list()
    for uID in IDs_n:
        if np.allclose(SS.N[uID], SS_n.N[uID]):
            candidateUIDs.append(uID)
    if len(candidateUIDs) > 0:
        SS.uIDs = np.arange(SS.K)
        SS_n.uIDs = np.arange(SS.K)
        Plan = dict(
            candidateUIDs=candidateUIDs,
            DTargetData=Data_n,
            targetSS=SS_n,
        )
        hmodel, SS, _, DResult = runDeleteMoveAndUpdateMemory(
            hmodel, SS, Plan,
            nRefineIters=2,
            LPkwargs=dict(limitMemoryLP=1),
            SSmemory=None,
        )
        if verbose >= 2:
            print('Cleanup deletes: %d/%d accepted' % (
                DResult['nAccept'], DResult['nTotal']))
        delattr(SS, 'uIDs')
        delattr(SS_n, 'uIDs')
        K_n -= DResult['nAccept']
    if verbose:
        K_true = len(np.unique(Data_n.TrueParams['Z']))
        print('Total states: %d' % (SS.K))
        print('Total states in cur seq: %d' % (K_n))
        print('Total true states in cur seq: %d' % (K_true))
    return hmodel, SS


def mergeDownLastStateIfPossible(resp_n, K_n, SS, SSprevComp, obsModel,
                                 verbose=False):
    ''' Try to merge the last state into the existing set.

    Returns
    -------
    resp_n : 2D array, size T x Kmax
    K_n : int
        total number of states used by sequence n
    SS_n : SuffStatBag
    '''
    if K_n == 1:
        return resp_n, K_n, SS

    kB = K_n - 1
    gaps = np.zeros(K_n - 1)
    for kA in range(K_n - 1):
        SS_kA = SS.getComp(kA, doCollapseK1=0)
        gaps[kA] = obsModel.calcHardMergeGap_SpecificPairSS(SS_kA, SSprevComp)
    kA = np.argmax(gaps)
    if gaps[kA] > -0.00001:
        if verbose >= 2:
            print("merging this block into state %d" % (kA))
        SS.mergeComps(kA, kB)
        mask = resp_n[:, kB] > 0
        resp_n[mask, kA] = 1
        resp_n[mask, kB] = 0
        return resp_n, K_n - 1, SS
    else:
        # No merge would be accepted
        return resp_n, K_n, SS


def getListOfContigBlocks(Data_n=None, initBlockLen=20, T=None):
    ''' Generate tuples identifying contiguous blocks within given seq.

    Examples
    --------
    >>> getListOfContigBlocks(T=31, initBlockLen=15)
    [(0, 15), (15, 30), (30, 31)]
    >>> getListOfContigBlocks(T=50, initBlockLen=20)
    [(0, 20), (20, 40), (40, 50)]
    >>> getListOfContigBlocks(T=20, initBlockLen=10)
    [(0, 10), (10, 20)]
    '''
    if T is None:
        start = Data_n.doc_range[0]
        stop = Data_n.doc_range[1]
        T = stop - start
    nBlocks = np.maximum(1, int(np.ceil(T / float(initBlockLen))))
    if nBlocks == 1:
        return [(0, T)]
    else:
        bList = list()
        for blockID in range(nBlocks):
            a = blockID * initBlockLen
            b = (blockID + 1) * initBlockLen
            if blockID == nBlocks - 1:
                b = T
            bList.append((a, b))
        return bList


"""

def proposeNewResp_randomSubwindowsOfContigBlocks(Z_n, propResp,
        origK=0,
        Kfresh=3,
        PRNG=np.random.RandomState,
        minBlockSize=0,
        maxBlockSize=10,
        **kwargs):
    ''' Create new value of resp matrix by randomly breaking up contig. blocks.

    Returns
    -------
    propResp : 2D array, N x K'
    '''
    propK = origK

    blockSizes, blockStarts = calcContigBlocksFromZ(Z_n)
    nBlocks = len(blockSizes)
    blockOrder = np.arange(nBlocks)
    PRNG.shuffle(blockOrder)
    for blockID in blockOrder:
        if blockSizes[blockID] <= minBlockSize:
            continue
        # Choose random subwindow of this block to create new state
        # min achievable size: minBlockSize + 1
        # max size: maxBlockSize
        maxBlockSize = np.minimum(maxBlockSize, blockSizes[blockID])
        wSize = PRNG.randint(minBlockSize, high=maxBlockSize)
        wStart = PRNG.randint(blockSizes[blockID] - wSize)
        wStart += blockStarts[blockID]
        wStop = wStart + wSize

        # Assign proposed window to new state
        propK = propK + 1
        propResp[wStart:wStop, :propK-1] = 0
        propResp[wStart:wStop, propK-1] = 1

        if propK == origK + Kfresh:
            break
    return propResp, propK
"""

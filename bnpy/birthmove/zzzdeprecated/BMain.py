import numpy as np
import bnpy.data

from BRefinery import makeCandidateLPWithNewComps
from BViz import showBirthBeforeAfter
import BLogger
from BirthProposalError import BirthProposalError

def runBirthMove(
        Data_b, curModel, curSS_notb, curLP_b,
        doVizBirth=0,
        **Plan):
    ''' Execute birth move on provided dataset and model

    Args
    -----
    Data_b : dataset
    hmodel : HModel object, with K comps
    SS_notb : SuffStatBag object with K comps
    LP_b : dict of local params, with K comps

    Returns
    -------
    Info : dict with fields
        LP_b : dict of local params for current batch, with K + Kx comps
    '''
    curModel = curModel.copy()
    propModel = curModel.copy()

    Plan['doVizBirth'] = doVizBirth

    # Create target dataset and LP
    try:
        Data_t, curLP_t, targetIDs = subsampleTargetFromDataAndLP(
            Data_b, curModel, curLP_b, **Plan)
    except BirthProposalError as e:
        # Planned target set does not exist, so exit
        BLogger.printEarlyExitMsg("Planned target dataset too small.", e)
        return dict(newLP_b=curLP_b, curLP_b=curLP_b)

    # Create relevant summaries
    curSS_b = curModel.get_global_suff_stats(
        Data_b, curLP_b, doPrecompEntropy=1)
    curSS_t = curModel.get_global_suff_stats(
        Data_t, curLP_t, doPrecompEntropy=1)
    curSS_nott = curSS_b - curSS_t

    if curSS_notb is not None:
        curSS_nott += curSS_notb

    # Log information about current target set
    relevantCompIDs = np.flatnonzero(curSS_t.getCountVec() > 0.01)
    relevantCompSizes = curSS_t.getCountVec()[relevantCompIDs]
    sortmask = np.argsort(-1*relevantCompSizes)
    relevantCompIDs = relevantCompIDs[sortmask]
    relevantCompSizes = relevantCompSizes[sortmask]
    Plan['relevantCompIDs'] = relevantCompIDs
    BLogger.printDataSummary(Data_t, curSS_t, curSS_nott, relevantCompIDs, Plan)

    # Evaluate current score
    curSS = curSS_nott + curSS_t
    curModel.update_global_params(curSS)
    curLscore = curModel.calc_evidence(SS=curSS)

    # Propose new local parameters for target set
    propLP_t, xcurSS_nott = makeCandidateLPWithNewComps(
        Data_t, curLP_t, propModel, curModel, curSS_nott, **Plan)
    # Evaluate proposal score
    propSS_t = propModel.get_global_suff_stats(
        Data_t, propLP_t, doPrecompEntropy=1)
    propSS = propSS_t + xcurSS_nott
    propModel.update_global_params(propSS)
    propLscore = propModel.calc_evidence(SS=propSS)

    propSize = propSS.getCountVec().sum()
    curSize = curSS.getCountVec().sum()
    assert np.allclose(propSize, curSize, rtol=0, atol=1e-6)

    if doVizBirth:
        showBirthBeforeAfter(**locals())
        keypress = input('Press any key to continue >>>')
        if keypress.count('embed'):
            from IPython import embed;
            embed()

    Result = dict(
        curLP_b=curLP_b,
        propLP_t=propLP_t,
        newLP_b=curLP_b,
        curModel=curModel,
        propModel=propModel,
        Lscore_gain=propLscore - curLscore,
        propLscore=propLscore,
        curLscore=curLscore,
        )
    if propLscore > curLscore:
        # Accept
        propLP_b = propModel.allocModel.fillSubsetLP(
            Data_b, curLP_b, propLP_t, targetIDs=targetIDs)
        Result['newLP_b'] = propLP_b
    else:
        # Reject. Do nothing.
        pass

    return Result

def subsampleTargetFromDataAndLP(Data, model, LP, **Plan):
    ''' Select target data items and associated local parameters.

    Returns
    -------
    Data_t : bnpy dataset
    curLP_t : dict of local params

    Post Condition
    --------------
    Plan : dict, with updated fields
    * targetIDs
    '''
    # Perform common parsing tasks
    btargetMaxSize = Plan['btargetMaxSize']
    if btargetMaxSize is None or btargetMaxSize == 'None':
        btargetMaxSize = Data.get_size()
    else:
        btargetMaxSize = int(btargetMaxSize)
    Plan['btargetMaxSize'] = btargetMaxSize
    btargetCompID = Plan['btargetCompID']
    if btargetCompID is None or btargetCompID == 'None':
        btargetCompID = None
    else:
        btargetCompID = int(btargetCompID)
    Plan['btargetCompID'] = btargetCompID
    
    # Subsample according to specific data object type
    if isinstance(Data, bnpy.data.BagOfWordsData):
        return _sample_target_BagOfWordsData(Data, model, LP, **Plan)
    elif isinstance(Data, bnpy.data.GroupXData):
        if model.getAllocModelName().count('Mixture'):
            Data = Data.toXData()
            return _sample_target_XData(Data, model, LP, **Plan)
        else:
            return _sample_target_GroupXData(Data, model, LP, **Plan)
    elif isinstance(Data, bnpy.data.XData):
        return _sample_target_XData(Data, model, LP, **Plan)


def _sample_target_XData(
        Data, model, LP, 
        PRNG=np.random, 
        btargetMaxSize=1000, btargetCompID=None, btargetRespThr=0.01, **Plan):
    ''' Select subset of provided XData dataset
    '''
    if btargetCompID is None:
        # For births based on current Data from batch
        size = np.minimum(Data.get_size(), btargetMaxSize)
        if size == Data.get_size():
            targetIDs = np.arange(size)
        else:
            targetIDs = PRNG.choice(
                Data.get_size(), size=btargetMaxSize, replace=False)
    else:
        targetIDs = np.flatnonzero(
            LP['resp'][:, btargetCompID] > btargetRespThr)
        if len(targetIDs) < 2:
            raise BirthProposalError('Target too small.')
        PRNG.shuffle(targetIDs)
        targetIDs = targetIDs[:btargetMaxSize]

    if len(targetIDs) < 2:
        raise BirthProposalError('Target too small.')
    Data_t = Data.select_subset_by_mask(
        targetIDs, doTrackFullSize=False, doTrackTruth=True)
    LP_t = model.allocModel.selectSubsetLP(Data, LP, targetIDs)
    Plan['targetIDs'] = targetIDs
    return Data_t, LP_t, targetIDs


def _sample_target_GroupXData(
        Data, model, LP, 
        PRNG=np.random, 
        btargetMaxSize=1000, btargetCompID=None, btargetRespThr=0.01, **Plan):
    ''' Select subset of provided GroupXData dataset
    '''
    if btargetCompID is None:
        # For births based on current Data from batch
        size = np.minimum(Data.get_size(), btargetMaxSize)
        if size == Data.get_size():
            targetIDs = np.arange(size)
        else:
            targetIDs = PRNG.choice(
                Data.get_size(), size=btargetMaxSize, replace=False)
    else:
        if model.getAllocModelName().count('TopicModel'):
            targetIDs = np.flatnonzero(
                LP['DocTopicCount'][:, btargetCompID] > btargetRespThr)
        elif model.getAllocModelName().count('HMM'):
            # select sequences that use the target state
            targetIDs = list()
            for n in range(Data.nDoc):
                N_n = np.sum(LP['resp'][:, btargetCompID], axis=0)
                if N_n > btargetRespThr:
                    targetIDs.append(n)
            targetIDs = np.asarray(targetIDs)
        else:
            targetIDs = np.flatnonzero(
                LP['resp'][:, btargetCompID] > btargetRespThr)
            Data = Data.toXData()
        if len(targetIDs) < 1:
            raise BirthProposalError('Target too small.')
        PRNG.shuffle(targetIDs)
        targetIDs = targetIDs[:btargetMaxSize]
    if len(targetIDs) < 1:
        raise BirthProposalError('Target too small.')

    Data_t = Data.select_subset_by_mask(
        targetIDs, doTrackFullSize=False, doTrackTruth=True)
    LP_t = model.allocModel.selectSubsetLP(Data, LP, targetIDs)
    Plan['targetIDs'] = targetIDs
    return Data_t, LP_t, targetIDs



"""
def _sample_target_BagOfWordsData(Data, model, LP, return_Info=0, **kwargs):
    ''' Get subsample of set of documents satisfying provided criteria.

    minimum size of each document, relationship to targeted component, etc.

    Keyword Args
    --------
    targetCompID : int, range: [0, 1, ... K-1]. **optional**
                 if present, we target documents that use a specific topic

    targetMinWordsPerDoc : int,
                         each document in returned targetData
                         must have at least this many words
    Returns
    --------
    targetData : BagOfWordsData dataset,
                with at most targetMaxSize documents
    DebugInfo : (optional), dictionary with debugging info
    '''
    DocWordMat = Data.getSparseDocTypeCountMatrix()
    DebugInfo = dict()

    candidates = np.arange(Data.nDoc)
    if kwargs['targetMinWordsPerDoc'] > 0:
        nWordPerDoc = np.asarray(DocWordMat.sum(axis=1))
        candidates = nWordPerDoc >= kwargs['targetMinWordsPerDoc']
        candidates = np.flatnonzero(candidates)
    if len(candidates) < 1:
        return None, dict()

    # ............................................... target a specific Comp
    if hasValidKey('targetCompID', kwargs):
        if hasValidKey('DocTopicCount', LP):
            Ndk = LP['DocTopicCount'][candidates].copy()
            Ndk /= np.sum(Ndk, axis=1)[:, np.newaxis] + 1e-9
            mask = Ndk[:, kwargs['targetCompID']] > kwargs['targetCompFrac']
        elif hasValidKey('resp', LP):
            mask = LP['resp'][
                :,
                kwargs['targetCompID']] > kwargs['targetCompFrac']
            if candidates is not None:
                mask = mask[candidates]
        else:
            raise ValueError('LP must have either DocTopicCount or resp')
        candidates = candidates[mask]

    # ............................................... target a specific Word
    elif hasValidKey('targetWordIDs', kwargs):
        wordIDs = kwargs['targetWordIDs']
        TinyMatrix = DocWordMat[candidates, :].toarray()[:, wordIDs]
        targetCountPerDoc = np.sum(TinyMatrix > 0, axis=1)
        mask = targetCountPerDoc >= kwargs['targetWordMinCount']
        candidates = candidates[mask]

    # ............................................... target based on WordFreq
    elif hasValidKey('targetWordFreq', kwargs):
        wordFreq = kwargs['targetWordFreq']
        from TargetPlannerWordFreq import calcDocWordUnderpredictionScores

        ScoreMat = calcDocWordUnderpredictionScores(Data, model, LP)
        ScoreMat = ScoreMat[candidates]
        DebugInfo['ScoreMat'] = ScoreMat
        if kwargs['targetSelectName'].count('score'):
            ScoreMat = np.maximum(0, ScoreMat)
            ScoreMat /= ScoreMat.sum(axis=1)[:, np.newaxis]
            distPerDoc = calcDistBetweenHist(ScoreMat, wordFreq)

            DebugInfo['distPerDoc'] = distPerDoc
        else:
            EmpWordFreq = DocWordMat[candidates, :].toarray()
            EmpWordFreq /= EmpWordFreq.sum(axis=1)[:, np.newaxis]
            distPerDoc = calcDistBetweenHist(EmpWordFreq, wordFreq)
            DebugInfo['distPerDoc'] = distPerDoc

        keepIDs = distPerDoc.argsort()[:kwargs['targetMaxSize']]
        candidates = candidates[keepIDs]
        DebugInfo['candidates'] = candidates
        DebugInfo['dist'] = distPerDoc[keepIDs]

    if len(candidates) < 1:
        return None, dict()
    elif len(candidates) <= kwargs['targetMaxSize']:
        targetData = Data.select_subset_by_mask(candidates)
    else:
        targetData = Data.get_random_sample(kwargs['targetMaxSize'],
                                            randstate=kwargs['randstate'],
                                            candidates=candidates)

    return targetData, DebugInfo
"""

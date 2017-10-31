'''
TargetDataSampler.py

Provides methods that sample target dataset

Sample selection criteria
---------
* targetMinNumWords (for bag-of-words data only)

'''
import numpy as np
import heapq
from scipy.spatial.distance import cdist

import bnpy.data


def add_to_ranked_target_data(RankedDataHeap, maxSize, Data, weights,
                              keep='largest'):
    '''
    '''
    docIDs = np.arange(Data.nDoc)

    # First, decide which docs are promising,
    #  since we don't want to blow-up memory costs by using *all* docs
    if len(RankedDataHeap) > 0:
        cutoffThr = RankedDataHeap[0][0]
        if keep == 'largest':
            docIDs = np.argsort(-1 * weights)[:maxSize]
            docIDs = docIDs[weights[docIDs] > cutoffThr]
        else:
            docIDs = np.argsort(weights)[:maxSize]
            docIDs = docIDs[weights[docIDs] < cutoffThr]

    if len(docIDs) < 1:
        return

    # For promising docs, convert to list-of-tuples format,
    #   and add to the heap
    if keep == 'largest':
        tList = Data.to_list_of_tuples(docIDs, w=weights)
    else:
        tList = Data.to_list_of_tuples(docIDs, w=-1 * weights)
    for docID, unitTuple in enumerate(tList):
        try:
            if len(RankedDataHeap) >= maxSize:
                heapq.heappushpop(RankedDataHeap, unitTuple)
            else:
                heapq.heappush(RankedDataHeap, unitTuple)
        except ValueError as error:
            # skip stupid errors related to duplicate weights
            pass

# sample_target_data
###########################################################


def sample_target_data(Data, model=None, LP=None, **kwargs):
    ''' Obtain subsample of provided dataset,

        Returns
        -------
        targetData : bnpy DataObj, with size at most kwargs['targetMaxSize']
    '''
    if isinstance(Data, bnpy.data.BagOfWordsData):
        return _sample_target_BagOfWordsData(Data, model, LP, **kwargs)
    elif isinstance(Data, bnpy.data.GroupXData):
        return _sample_target_GroupXData(Data, model, LP, **kwargs)
    elif isinstance(Data, bnpy.data.XData):
        return _sample_target_XData(Data, model, LP, **kwargs)


def _sample_target_GroupXData(Data, model, LP, **kwargs):
    ''' Draw sample subset of provided GroupXData dataset
    '''
    randstate = kwargs['randstate']
    if not hasValidKey('targetCompID', kwargs):
        raise NotImplementedError('TODO')

    ktarget = kwargs['targetCompID']
    targetProbThr = kwargs['targetCompFrac']
    mask = LP['resp'][:, ktarget] > targetProbThr
    objIDs = np.flatnonzero(mask)
    if len(objIDs) < 2:
        return None, dict()
    randstate.shuffle(objIDs)
    targetObjIDs = objIDs[:kwargs['targetMaxSize']]
    TargetData = Data.select_subset_by_mask(atomMask=targetObjIDs,
                                            doTrackFullSize=False)
    TargetInfo = dict(ktarget=ktarget)
    return TargetData, TargetInfo


def _sample_target_XData(Data, model, LP, **kwargs):
    ''' Draw sample subset of provided XData dataset
    '''
    randstate = kwargs['randstate']
    if hasValidKey('targetCompID', kwargs):
        ktarget = kwargs['targetCompID']
        targetProbThr = kwargs['targetCompFrac']
        mask = LP['resp'][:, ktarget] > targetProbThr
        objIDs = np.flatnonzero(mask)
        if len(objIDs) < 2:
            return None, dict()
        randstate.shuffle(objIDs)
        targetObjIDs = objIDs[:kwargs['targetMaxSize']]
        TargetData = Data.select_subset_by_mask(targetObjIDs,
                                                doTrackFullSize=False)
        TargetInfo = dict(ktarget=ktarget)
    else:
        # For births based on current Data from batch
        size = np.minimum(Data.get_size(), kwargs['targetMaxSize'])
        if size == Data.get_size():
            TargetData = Data
        else:
            targetIDs = list(range(Data.get_size()))
            randstate.shuffle(targetIDs)
            TargetData = Data.select_subset_by_mask(targetIDs[:size],
                                                    doTrackFullSize=False)
        TargetInfo = dict(msg='batch birth')
    return TargetData, TargetInfo


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


def hasValidKey(key, kwargs):
    return key in kwargs and kwargs[key] is not None


def makeHeldoutData(targetData, **kwargs):
    nDoc = targetData.nDoc
    nHoldout = nDoc / 5
    holdIDs = kwargs['randstate'].choice(nDoc, nHoldout, replace=False)
    trainIDs = [x for x in range(nDoc) if x not in holdIDs]
    holdData = targetData.select_subset_by_mask(docMask=holdIDs,
                                                doTrackFullSize=False)
    targetData = targetData.select_subset_by_mask(docMask=trainIDs,
                                                  doTrackFullSize=False)
    return targetData, holdData


def calcDistBetweenHist(Xfreq, yfreq, targetDistMethod='intersection'):
    if targetDistMethod == 'intersection':
        return 1 - np.sum(np.minimum(Xfreq, yfreq), axis=1)
    raise NotImplementedError('UNKNOWN: ' + targetDistMethod)


def calcKLdivergence_discrete(P1, P2):
    KL = np.log(P1 + 1e-100) - np.log(P2 + 1e-100)
    KL *= P1
    return KL.sum(axis=1)


def getDataExemplar(Data):
    ''' Return 'exemplar' for this dataset
    '''
    if Data is None:
        return 0
    start = 0
    stop = Data.doc_range[0, 1]
    wordFreq = np.zeros(Data.vocab_size)
    wordFreq[Data.word_id[start:stop]] = Data.word_count[start:stop]
    return wordFreq / wordFreq.sum()


def getSize(Data):
    ''' Return the integer size of the provided dataset
    '''
    if Data is None:
        return 0
    elif hasattr(Data, 'nDoc'):
        return Data.nDoc
    else:
        return Data.nObs

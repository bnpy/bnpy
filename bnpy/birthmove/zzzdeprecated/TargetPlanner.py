'''
TargetPlanner.py

Advanced selection of plan-of-attack for improving current model via birthmove.

Key methods
--------
  * select_target_comp
'''
import numpy as np
from collections import defaultdict

from BirthProposalError import BirthProposalError
import BirthLogger

EPS = 1e-14


def makePlans_TargetCompsSmart(SS, BirthRecordsByComp, lapFrac,
                               ampF=1.0,
                               **birthKwArgs):
    ''' Determine which components to target, via tracked previous attempts.

        Returns
        --------
        Plans : list of Plan dicts, with fields
        * ktarget : position of chosen comp in current order 0, 1, ... K-1
        * targetUID : unique ID (like in ActiveIDVec or SS.uIDs)
        * count : size of chosen comp at moment of choosing in SS.getCountVec

        Updates (in place)
        --------
        BirthRecordsByComp : some entries may be removed
        Removes comps reactivated to eligible list (all failures forgotten),
        which happens if it has changed size exceeds prescribed limit
    '''
    nBirths = birthKwArgs['birthPerLap']
    MIN_PERC_DIFF = birthKwArgs['birthChangeInSizeToReactivate']
    MAX_FAIL = birthKwArgs['birthFailLimit']

    eligibleIDs = list()
    eligibleSizes = list()
    waitListIDs = list()
    waitListSizes = list()

    Nvec = SS.getCountVec() * ampF
    for k, compID in enumerate(SS.uIDs):
        if Nvec[k] < birthKwArgs['targetMinSize']:
            continue
        if compID in BirthRecordsByComp:
            if 'nFail' in BirthRecordsByComp[compID]:
                nFails = BirthRecordsByComp[compID]['nFail']
                prevN_k = BirthRecordsByComp[compID]['count']

                percDiff = np.abs(Nvec[k] - prevN_k) / (1e-8 + Nvec[k])
                if percDiff < MIN_PERC_DIFF:
                    if nFails < MAX_FAIL:
                        waitListIDs.append(compID)
                        waitListSizes.append(Nvec[k])
                    continue
                else:
                    # Comp has changed size enough to warrant reactivation
                    # So, we take it off the "disabled/failure" list
                    del BirthRecordsByComp[compID]
                    msg = 'reactivating \
                        comp %d. newN %.1f, oldN %.1f, percDiff %.3f' \
                        % (compID, Nvec[k], prevN_k, percDiff)
                    BirthLogger.log(msg, level='debug')

        eligibleIDs.append(compID)
        eligibleSizes.append(Nvec[k])

    if len(eligibleIDs) < nBirths and len(waitListIDs) > 0:
        nExtra = nBirths - len(eligibleIDs)
        # Add in comps that have failed before
        # prioritizing largest options
        order = np.argsort(waitListSizes)[-nExtra:]
        eligibleIDs.extend([waitListIDs[x] for x in order])
        eligibleSizes.extend([waitListSizes[x] for x in order])

    # Greedy selection of the top nBirths eligibles
    if len(eligibleIDs) > nBirths:
        eligibleIDs = np.asarray(eligibleIDs)
        eligibleSizes = np.asarray(eligibleSizes)
        order = np.argsort(-1 * eligibleSizes)[:nBirths]
        eligibleIDs = [eligibleIDs[x] for x in order]
        eligibleSizes = [eligibleSizes[x] for x in order]

    msgList = ['%d:%.0f' % (eligibleIDs[i], eligibleSizes[i])
               for i in range(len(eligibleIDs))]
    msg = 'Top Eligibles: ' + ' '.join(msgList)
    BirthLogger.log(msg, level='moreinfo')

    Plans = list()
    for birthID, compID in enumerate(eligibleIDs):
        Plan = dict(Data=None,
                    targetWordIDs=None,
                    targetWordFreq=None)
        ktarget = np.flatnonzero(compID == SS.uIDs)
        assert ktarget.size == 1
        ktarget = int(ktarget[0])
        Plan['ktarget'] = ktarget
        Plan['targetUID'] = compID
        Plan['count'] = Nvec[ktarget]
        Plans.append(Plan)

    return Plans


def makePlansToTargetWordFreq(model=None, LP=None, Data=None, Q=None,
                              targetSelectName='anchorFreq', nPlans=1,
                              **kwargs):
    K = model.obsModel.K
    topics = np.zeros((K, Q.shape[1]))
    for k in range(K):
        topics[k, :] = model.obsModel.comp[k].lamvec
        topics[k, :] = topics[k, :] / topics[k, :].sum()
    Q = Q / Q.sum(axis=1)[:, np.newaxis]
    choices = GramSchmidtUtil.FindAnchorsForExpandedBasis(Q, topics,
                                                          nPlans)
    Plans = list()
    for pID, choice in enumerate(choices):
        Plan = dict(ktarget=None, Data=None, targetWordIDs=None,
                    targetWordFreq=Q[choice])
        topWords = np.argsort(-1 * Plan['targetWordFreq'])[:10]
        if hasattr(Data, 'vocab_dict'):
            Vocab = getVocab(Data)
            anchorWordStr = 'Anchor: ' + Vocab[choice]
            topWordStr = ' '.join([Vocab[w] for w in topWords])
        else:
            anchorWordStr = 'Anchor: ' + str(choice)
            topWordStr = ' '.join([str(w) for w in topWords])
        Plan['anchorWordID'] = choice
        Plan['topWordIDs'] = topWords
        Plan['log'] = list()
        Plan['log'].append(anchorWordStr)
        Plan['log'].append(topWordStr)
        Plans.append(Plan)
    return Plans


def getVocab(Data):
    return [str(x[0][0]) for x in Data.vocab_dict]


def select_target_words_MultipleSets(model=None,
                                     LP=None, Data=None, SS=None, Q=None,
                                     nSets=1, targetNumWords=10, **kwargs):
    goodWords = get_good_words(Data)
    DocWordFreq_emp = calcWordFreqPerDoc_empirical(Data)
    DocWordFreq_model = calcWordFreqPerDoc_model(model, LP)
    DocWordFreq_emp = DocWordFreq_emp[:, goodWords]
    DocWordFreq_model = DocWordFreq_model[:, goodWords]

    uError = np.maximum(DocWordFreq_emp - DocWordFreq_model, 0)
    uErrorPerDoc = np.sum(uError, axis=1)

    # Find two similar docs that have very large underprediction error
    mostUnexplainedDocs = np.argsort(-1 * uErrorPerDoc)[:200]
    uErrBiggest = uError[mostUnexplainedDocs]
    binErrBig = uErrBiggest > 0

    # Find the two most similar docs among this top list
    from scipy.spatial.distance import cdist
    D = cdist(uErrBiggest, uErrBiggest, 'cityblock')
    D[D == 0] = Data.vocab_size  # make bad pairs impossible to pick

    nSets = np.minimum(nSets, D.shape[0])
    Plans = list()
    for x in range(nSets):
        bestPair = np.argmin(D.flatten())
        a, b = np.unravel_index(bestPair, D.shape)

        # eliminate these docs from further consideration
        D[a, :] = Data.vocab_size
        D[:, a] = Data.vocab_size
        D[:, b] = Data.vocab_size
        D[b, :] = Data.vocab_size
        score = uErrBiggest[a] + uErrBiggest[b]
        onTopicWords = goodWords[np.argsort(-1 * score)[:targetNumWords]]
        if hasattr(Data, 'vocab_dict'):
            Vocab = [str(x[0][0]) for x in Data.vocab_dict]
            print('Anchor Doc %d' % (a))
            print(' '.join([Vocab[goodWords[w]] for w in np.argsort(
                -1 * uErrBiggest[a])[:20]]))
            print('Anchor Doc %d' % (b))
            print(' '.join([Vocab[goodWords[w]] for w in np.argsort(
                -1 * uErrBiggest[b])[:20]]))
            print('BOTH')
            print(' '.join([Vocab[w] for w in onTopicWords]))
        Plans.append(dict(targetWordIDs=onTopicWords, ktarget=None, Data=None))
    return Plans


def select_target_comp(K, SS=None, model=None, LP=None, Data=None,
                       lapsSinceLastBirth=defaultdict(int),
                       excludeList=list(), doVerbose=False, return_ps=False,
                       **kwargs):
    ''' Choose a single component among possible choices {0,1,2, ... K-2, K-1}
        to target with a birth proposal.

        Keyword Args
        -------
        randstate : numpy RandomState object, allows random choice of ktarget
        targetSelectName : string, identifies procedure for selecting ktarget
                            {'uniform','sizebiased', 'delaybiased',
                             'delayandsizebiased'}

        Returns
        -------
        ktarget : int, range: 0, 1, ... K-1
                  identifies single component in the current model to target

        Raises
        -------
        BirthProposalError, if cannot select a valid choice
    '''
    targetSelectName = kwargs['targetSelectName']
    if SS is None:
        targetSelectName = 'uniform'
        assert K is not None
    else:
        assert K == SS.K

    if len(excludeList) >= K:
        msg = 'BIRTH not possible. All K=%d targets used or excluded.' % (K)
        raise BirthProposalError(msg)

    # Build vector ps: gives probability of each choice
    ########
    ps = np.zeros(K)
    if targetSelectName == 'uniform':
        ps = np.ones(K)
    elif targetSelectName == 'sizebiased':
        ps = SS.N.copy()
    elif targetSelectName == 'delaybiased':
        # Bias choice towards components that have not been selected in a long
        # time
        lapDist = np.asarray([lapsSinceLastBirth[kk] for kk in range(K)])
        ps = np.maximum(lapDist + 1e-5, 0)
        ps = ps * ps
    elif targetSelectName == 'delayandsizebiased':
        # Bias choice towards comps that have not been selected in a long time
        #  *and* which have many members
        lapDist = np.asarray([lapsSinceLastBirth[kk] for kk in range(K)])
        ps = np.maximum(lapDist + 1e-5, 0)
        ps = ps * ps * SS.N
    elif targetSelectName == 'predictionQuality':
        ps = calc_underprediction_scores_per_topic(K, model, Data, LP,
                                                   excludeList, **kwargs)
        ps = ps * ps  # make more peaked!
    else:
        raise NotImplementedError(
            'Unrecognized procedure: ' +
            targetSelectName)
    if SS is not None:
        ps[SS.N < kwargs['targetMinSize']] = 0

    # Make a choice using vector ps, if possible. Otherwise, raise error.
    ########
    ps[excludeList] = 0
    if np.sum(ps) < EPS:
        msg = 'BIRTH not possible. All K=%d targets too small or excluded.' % (
            K)
        raise BirthProposalError(msg)
    ps = ps / np.sum(ps)
    assert np.allclose(np.sum(ps), 1.0)

    ktarget = kwargs['randstate'].choice(K, p=ps)
    if return_ps:
        return ktarget, ps
    return ktarget


def select_target_words(model=None,
                        LP=None, Data=None, SS=None, Q=None,
                        excludeList=list(), doVerbose=False, return_ps=0,
                        **kwargs):
    ''' Choose a set of vocabulary words to target with a birth proposal.

        Keyword Args
        -------
        randstate : numpy RandomState object, allows random choice of relWords
        targetSelectName : string, identifies procedure for selecting ktarget
                            {'uniform','predictionQuality'}

        Returns
        -------
        relWords : list, each entry in {0, 1, ... Data.vocab_size-1}

        Raises
        -------
        BirthProposalError, if cannot select a valid choice
    '''
    nWords = kwargs['targetNumWords']
    targetSelectName = kwargs['targetSelectName']
    if targetSelectName == 'wordUniform':
        pWords = np.ones(Data.vocab_size)
        doRelWords = 0
    elif targetSelectName == 'wordPredictionQuality':
        pWords = calc_underprediction_scores_per_word(model,
                                                      Data, LP=LP, **kwargs)
        doRelWords = 1
    elif targetSelectName == 'wordQualityClosestPair':
        pWords = calc_word_scores_for_closest_pair_of_underpredicted_docs(
            model, Data, LP, **kwargs)
        doRelWords = 0
    elif targetSelectName == 'anchorWords':
        assert Q is not None
        pWords = calc_word_scores_for_anchor(model, Q, Data, **kwargs)
        doRelWords = 0
    else:
        raise NotImplementedError(
            'Unrecognized procedure: ' +
            targetSelectName)
    pWords[excludeList] = 0
    if np.sum(pWords) < EPS:
        msg = 'BIRTH not possible. All words have zero probability.'
        raise BirthProposalError(msg)

    if doRelWords:
        words = sample_related_words_by_score(Data, pWords, nWords=nWords,
                                              **kwargs)
    else:
        nWords = np.minimum(nWords, (pWords > EPS).sum())
        words = np.argsort(-1 * pWords)[:nWords]
    if words is None or len(words) < 1:
        msg = 'BIRTH not possible. Word selection failed.'
        raise BirthProposalError(msg)

    if hasattr(Data, 'vocab_dict'):
        Vocab = [str(x[0][0]) for x in Data.vocab_dict]
        print('TARGETED WORDS')
        print(' '.join([Vocab[w] for w in words]))

    if return_ps:
        return words, pWords
    return words


def calc_word_scores_for_anchor(model, Q, Data, **kwargs):
    goodWords = get_good_words(Data)
    K = model.obsModel.K
    topics = np.zeros((K, Q.shape[1]))
    for k in range(K):
        topics[k, :] = model.obsModel.comp[k].lamvec
        topics[k, :] = topics[k, :] / topics[k, :].sum()
    # Normalization happens internally in this function
    choices = GramSchmidtUtil.FindAnchorsForExpandedBasis(Q[goodWords], topics,
                                                          10)

    choice = kwargs['randstate'].choice(choices, 1)
    chosenWord = goodWords[choice]
    if hasattr(Data, 'vocab_dict'):
        Vocab = [str(x[0][0]) for x in Data.vocab_dict]
        print('CHOSEN ANCHOR WORD: %s' % (Vocab[chosenWord]))
    score = np.zeros(Data.vocab_size)
    score[goodWords] = Q[chosenWord, goodWords]

    # make more peaked!
    score = score * score
    score /= score.sum()
    return score


def calc_underprediction_scores_per_topic(K, model, Data, LP=None,
                                          excludeList=list(), **kwargs):
    ''' Calculate for each topic a scalar weight. Larger => worse prediction.
    '''
    if str(type(model.allocModel)).count('HDP') > 0:
        return _hdp_calc_underprediction_scores(
            K, model, Data, LP, excludeList, **kwargs)
    else:
        return _dp_calc_underprediction_scores(
            K, model, Data, LP, excludeList, **kwargs)


def _dp_calc_underprediction_scores(K, model, Data, LP, xList, **kwargs):
    ''' Calculate for each topic a scalar weight. Larger => worse prediction.
    '''
    if LP is None:
        LP = model.calc_local_params(Data)
    assert K == model.allocModel.K

    # Empirical word frequencies (only for docs with enough words)
    empWordFreq = Data.to_sparse_docword_matrix()
    Nd = np.squeeze(np.asarray(empWordFreq.sum(axis=1)))
    candidateDocs = np.flatnonzero(Nd > kwargs['targetMinWordsPerDoc'])
    empWordFreq = empWordFreq[candidateDocs].toarray()
    empWordFreq /= empWordFreq.sum(axis=1)[:, np.newaxis]
    resp = LP['resp'][candidateDocs]

    # Compare to model's expected frequencies
    score = np.zeros(K)
    for k in range(K):
        if k in xList:
            continue
        lamvec = model.obsModel.comp[k].lamvec
        modelWordFreq_k = lamvec / lamvec.sum()
        for d in np.flatnonzero(resp[:, k] > kwargs['targetCompFrac']):
            score[k] += resp[d, k] * \
                calcKL_discrete(empWordFreq, modelWordFreq_k)
    score = np.maximum(score, 0)
    score /= score.sum()
    return score


def _hdp_calc_underprediction_scores(K, model, Data, LP, xList, **kwargs):
    ''' Calculate for each topic a scalar weight. Larger => worse prediction.
    '''
    if LP is None or 'word_variational' not in LP:
        LP = model.calc_local_params(Data, LP, methodLP='memo')
    assert K == model.allocModel.K
    NdkThr = 1.0 / K * kwargs['targetMinWordsPerDoc']
    score = np.zeros(K)
    for k in range(K):
        if k in xList:
            continue
        lamvec = model.obsModel.comp[k].lamvec
        modelWordFreq_k = lamvec / lamvec.sum()
        candidateDocs = np.flatnonzero(LP['DocTopicCount'][:, k] > NdkThr)
        for d in candidateDocs:
            start = Data.doc_range[d, 0]
            stop = Data.doc_range[d, 1]
            word_id = Data.word_id[start:stop]
            word_count = Data.word_count[start:stop]
            resp = LP['word_variational'][start:stop, k]
            empWordFreq_k = np.zeros(Data.vocab_size)
            empWordFreq_k[word_id] = (resp * word_count)
            empWordFreq_k = empWordFreq_k / empWordFreq_k.sum()
            if 'theta' in LP:
                probInDoc_k = LP['theta'][d, k] / LP['theta'][d, :].sum()
            else:
                probInDoc_k = LP['U1'][d, k] * np.prod(1.0 - LP['U0'][d, :k])
            score[k] += probInDoc_k * calcKL_discrete(empWordFreq_k,
                                                      modelWordFreq_k)
    # Make score a valid probability vector
    score = np.maximum(score, 0)
    score /= score.sum()
    return score


def get_good_words(Data):
    nDPerWord = Data.getNumDocsPerWord()
    UpLimit = 0.15 * Data.nDoc
    isToyData = int(np.sqrt(Data.vocab_size)) == np.sqrt(Data.vocab_size)
    if isToyData:
        UpLimit = Data.nDoc  # only toy data hits this line
    goodWords = np.flatnonzero((nDPerWord < UpLimit) * (nDPerWord > 25))
    return goodWords


def calc_word_scores_for_closest_pair_of_underpredicted_docs(
        model, Data, LP, **kwargs):
    goodWords = get_good_words(Data)

    DocWordFreq_emp = calcWordFreqPerDoc_empirical(Data)
    DocWordFreq_model = calcWordFreqPerDoc_model(model, LP)

    DocWordFreq_emp = DocWordFreq_emp[:, goodWords]
    DocWordFreq_model = DocWordFreq_model[:, goodWords]

    uError = np.maximum(DocWordFreq_emp - DocWordFreq_model, 0)
    uErrorPerDoc = np.sum(uError, axis=1)

    # Find two similar docs that have very large underprediction error
    mostUnexplainedDocs = np.argsort(-1 * uErrorPerDoc)[:150]
    uErrBiggest = uError[mostUnexplainedDocs]

    binErrBig = uErrBiggest > 0

    # Find the two most similar docs among this top list
    from scipy.spatial.distance import cdist
    D = cdist(uErrBiggest, uErrBiggest, 'cityblock')
    D[D == 0] = Data.vocab_size  # make bad pairs impossible to pick

    # Narrow down to the top ten pairs
    bestPairs = np.argsort(D.flatten())[::2][:10]
    bestIndex = kwargs['randstate'].choice(bestPairs)

    i, j = np.unravel_index(bestIndex, D.shape)

    # Identify words that are highly underpredicted in both documents
    score = np.zeros(Data.vocab_size)
    score[goodWords] = uErrBiggest[i] + uErrBiggest[j]
    score = np.maximum(score, 0)
    score = score ** 2  # make more peaked!
    score /= score.sum()
    if hasattr(Data, 'vocab_dict'):
        Vocab = [str(x[0][0]) for x in Data.vocab_dict]
        print('Anchor Doc 1')
        print(' '.join([Vocab[goodWords[w]] for w in np.argsort(
            -1 * uErrBiggest[i])[:20]]))
        print('Anchor Doc 2')
        print(' '.join([Vocab[goodWords[w]] for w in np.argsort(
            -1 * uErrBiggest[j])[:20]]))
        print('BOTH')
        print(' '.join([Vocab[w] for w in np.argsort(-1 * score)[:20]]))
    return score


def calc_underprediction_scores_per_word(model, Data, LP=None, **kwargs):
    ''' Find scalar score for each vocab word. Larger => worse prediction.
    '''
    if LP is None:
        LP = model.calc_local_params(Data)
    DocWordFreq_emp = calcWordFreqPerDoc_empirical(Data)
    DocWordFreq_model = calcWordFreqPerDoc_model(model, LP)
    uError = np.maximum(DocWordFreq_emp - DocWordFreq_model, 0)
    # For each word, identify set of relevant documents
    DocWordMat = Data.to_sparse_docword_matrix().toarray()
    score = np.zeros(Data.vocab_size)
    # TODO: only consider words with many docs overall
    for vID in range(Data.vocab_size):
        countPerDoc = DocWordMat[:, vID]
        typicalWordCount = np.median(countPerDoc[countPerDoc > 0])
        candidateDocs = np.flatnonzero(countPerDoc > typicalWordCount)
        if len(candidateDocs) < 10:
            continue
        score[vID] = np.mean(uError[candidateDocs, vID])
    # Only give positive probability to words with above average score
    score = score - np.mean(score)
    score = np.maximum(score, 0)
    score = score * score  # make more peaked!
    score /= score.sum()
    return score


def sample_related_words_by_score(Data, pscore, nWords=3, anchor=None,
                                  doVerbose=False,
                                  **kwargs):
    ''' Sample set of words that have high underprediction score AND cooccur.
    '''
    DocWordArr = Data.to_sparse_docword_matrix().toarray()
    Cov = np.cov(DocWordArr.T, bias=1)
    sigs = np.sqrt(np.diag(Cov))
    Corr = Cov / np.maximum(np.outer(sigs, sigs), 1e-10)
    posCorr = np.maximum(Corr, 0)
    assert not np.any(np.isnan(posCorr))

    randstate = kwargs['randstate']
    if anchor is None:
        anchors = randstate.choice(
            Data.vocab_size,
            nWords,
            replace=False,
            p=pscore)
    else:
        anchors = [anchor]
    for firstWord in anchors:
        curWords = [firstWord]
        while len(curWords) < nWords:
            relWordProbs = calc_prob_related_words(posCorr, pscore, curWords)
            if np.sum(relWordProbs) < 1e-14 or np.any(np.isnan(relWordProbs)):
                break
            newWord = randstate.choice(Data.vocab_size, 1, replace=False,
                                       p=relWordProbs)
            curWords.append(int(newWord))
            if doVerbose:
                print(curWords)
        if len(curWords) == nWords:
            return curWords
    return anchors


def calc_prob_related_words(posCorr, pscore, curWords):
    relWordProbs = np.prod(posCorr[curWords, :], axis=0) * pscore
    relWordProbs[curWords] = 0
    relWordProbs = np.maximum(relWordProbs, 0)
    relWordProbs /= relWordProbs.sum()
    return relWordProbs


def calcWordFreqPerDoc_empirical(Data, candidateDocs=None):
    ''' Build empirical distribution over words for each document.
    '''
    DocWordMat = Data.to_sparse_docword_matrix()
    if candidateDocs is None:
        DocWordFreq_emp = DocWordMat.toarray()
    else:
        DocWordFreq_emp = DocWordMat[candidateDocs].toarray()
    DocWordFreq_emp /= 1e-9 + DocWordFreq_emp.sum(axis=1)[:, np.newaxis]
    return DocWordFreq_emp


def calcWordFreqPerDoc_model(model, LP, candidateDocs=None):
    ''' Build model's expected word distribution for each document
    '''
    if candidateDocs is None:
        Prior = np.exp(LP['E_logPi'])  # D x K
    else:
        Prior = np.exp(LP['E_logPi'][candidateDocs])  # D x K
    Lik = np.exp(model.obsModel.getElogphiMatrix())  # K x V
    DocWordFreq_model = np.dot(Prior, Lik)
    DocWordFreq_model /= DocWordFreq_model.sum(axis=1)[:, np.newaxis]
    return DocWordFreq_model


def calcKL_discrete(P1, P2):
    KL = np.log(P1 + 1e-100) - np.log(P2 + 1e-100)
    KL *= P1
    return KL.sum()

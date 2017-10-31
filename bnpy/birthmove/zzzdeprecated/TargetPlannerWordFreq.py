'''
TargetPlannerWordFreq.py
'''
import numpy as np
from collections import defaultdict

from BirthProposalError import BirthProposalError

EPS = 1e-14


def MakePlans(Data, model, LP, Q=None, **kwargs):
    ''' Create list of Plans
    '''
    newTopics, Info = makeCandidateTopics(Data, Q, model, LP, **kwargs)
    if 'doVizBirth' in kwargs and kwargs['doVizBirth']:
        from matplotlib import pylab
        pylab.imshow(newTopics, vmin=0, vmax=0.01,
                     aspect=Data.vocab_size / newTopics.shape[0],
                     interpolation='nearest')
    Plans = list()
    for kk in range(newTopics.shape[0]):
        Plan = dict(ktarget=None, Data=None, targetWordIDs=None,
                    targetWordFreq=newTopics[kk])

        # Add material to the log
        topWords = np.argsort(-1 * Plan['targetWordFreq'])[:10]
        if hasattr(Data, 'vocab_dict'):
            Vocab = getVocab(Data)
            topWordStr = ' '.join([Vocab[w] for w in topWords])
        else:
            topWordStr = ' '.join([str(w) for w in topWords])
        Plan['log'] = list()
        Plan['topWordIDs'] = topWords
        Plan['log'].append(topWordStr)

        if 'anchorID' in Info:
            anchorWordStr = 'Anchor: ' + str(Info['anchorID'][kk])
            Plan['anchorWordID'] = anchorWordStr
            Plan['log'].append(anchorWordStr)
        Plans.append(Plan)
    return Plans


def makeCandidateTopics(Data, Q, model, LP, **kwargs):
    '''
    '''
    selectName = kwargs['targetSelectName']
    if Q is None:
        if hasattr(Data, 'Q'):
            Q = Data.Q
        else:
            # fall back on this routine
            selectName = 'missingTopic'
    if selectName.count('anchor'):
        return makeCandidateTopics_AnchorAnalysis(Q, model, **kwargs)
    elif selectName.count('missingTopic'):
        return makeCandidateTopics_MissingTopic(Data, model, LP, **kwargs)
    else:
        raise NotImplementedError('UNKNOWN: ' + selectName)


# Useful helpers
###########################################################
def getTopicWordFreqFromModel(model):
    K = model.obsModel.K
    topics = np.zeros((K, model.obsModel.comp[0].D))
    for k in range(K):
        topics[k, :] = model.obsModel.comp[k].lamvec
        topics[k, :] = topics[k, :] / topics[k, :].sum()
    return topics


def getVocab(Data):
    return [str(x[0][0]) for x in Data.vocab_dict]

# Spectral analysis
###########################################################


def makeCandidateTopics_AnchorAnalysis(Q, model, nPlans=1,
                                       seed=0, **kwargs):
    ''' Returns matrix of candidate topic-word parameters for each of n Plans
    '''
    Q = Q / Q.sum(axis=1)[:, np.newaxis]

    # Find few rows of Q that best explain *all* rows of Q (together with
    # curTopics)
    curTopics = getTopicWordFreqFromModel(model)
    choices = GramSchmidtUtil.FindAnchorsForExpandedBasis(Q, curTopics,
                                                          nPlans)
    newTopics = Q[choices]
    return newTopics, dict(anchorID=choices)

# Underprediction scores
###########################################################


def makeCandidateTopics_MissingTopic(Data, model, LP, nPlans=1,
                                     seed=0, **kwargs):
    ''' Returns matrix of candidate topic-word parameters for each of n Plans
    '''
    import KMeansRex
    ScoreMat = calcDocWordUnderpredictionScores(Data, model, LP)

    # Condense to one scalar per doc,
    #  where higher value => more distance between Model and Emp
    AScorePerDoc = np.sum(np.abs(ScoreMat), axis=1)

    # Keep only the worst 10% of all docs,
    nKeep = np.maximum(int(0.1 * Data.nDoc), 50)
    keepIDs = np.argsort(-1 * AScorePerDoc)[:nKeep]

    ScoreMat = ScoreMat[keepIDs]
    ScoreMat = np.maximum(0, ScoreMat)
    ScoreMat /= ScoreMat.sum(axis=1)[:, np.newaxis]

    K = np.maximum(nPlans, 2)
    newTopics, Z = KMeansRex.RunKMeans(ScoreMat, K,
                                       initname='plusplus',
                                       Niter=10, seed=seed)

    # Remove uncommon candidates (based on very little data)
    Nk, binedges = np.histogram(np.squeeze(Z), np.arange(-0.5, K))
    newTopics = newTopics[Nk >= 5]
    Nk = Nk[Nk >= 5]

    if nPlans < newTopics.shape[0]:
        newTopics = newTopics[np.argmax(Nk)]

    if newTopics.ndim == 1:
        newTopics = newTopics[np.newaxis, :]

    # Normalize and return
    newTopics = np.maximum(1e-8, newTopics)
    newTopics /= newTopics.sum(axis=1)[:, np.newaxis]
    return newTopics, dict(Nk=Nk)


def calcDocWordUnderpredictionScores(Data, model, LP):
    ''' Calculate D x V matrix, where larger values => underprediction
    '''
    WordFreq_model = calcWordFreq_model(model, LP)
    WordFreq_emp = calcWordFreq_empirical(Data)
    DocWordScores = WordFreq_emp - WordFreq_model
    return DocWordScores


def calcWordFreq_model(model, LP):
    Prior = np.exp(LP['E_logPi'])
    Lik = model.obsModel.getElogphiMatrix()
    Lik = np.exp(Lik - Lik.max(axis=1)[:, np.newaxis])
    WordFreq_model = np.dot(Prior, Lik)
    WordFreq_model /= WordFreq_model.sum(axis=1)[:, np.newaxis]
    return WordFreq_model


def calcWordFreq_empirical(Data):
    WordFreq_emp = Data.to_sparse_docword_matrix().toarray()
    WordFreq_emp /= WordFreq_emp.sum(axis=1)[:, np.newaxis]
    return WordFreq_emp

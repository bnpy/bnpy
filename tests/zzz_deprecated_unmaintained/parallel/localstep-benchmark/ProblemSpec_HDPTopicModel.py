import numpy as np
import bnpy

def makeData(K=10, vocab_size=5000, nDocTotal=10, 
             nWordsPerDoc=200,
             **kwargs):
    ''' Create bag-of-words toy dataset for topic modeling
    '''
    PRNG = np.random.RandomState(0)
    topics = PRNG.gamma(0.5, 1, size=(K, vocab_size))
    topics /= topics.sum(axis=1)[:,np.newaxis]
    topic_prior = 0.5 * np.ones(K)
    Data = bnpy.data.BagOfWordsData.CreateToyDataFromLDAModel(
        nDocTotal=nDocTotal,
        nWordsPerDoc=nWordsPerDoc,
        K=K,
        topic_prior=topic_prior,
        topics=topics)
    return Data

def makeModel(Data=None, K=10, **kwargs):
    # Create model and initialize global parameters
    APriorSpec = dict(gamma=10, alpha=0.5)
    OPriorSpec = dict(lam=0.5)
    hmodel = bnpy.HModel.CreateEntireModel(
        'VB', 'HDPTopicModel', 'Mult', APriorSpec, OPriorSpec,
        Data=Data)
    hmodel.init_global_params(Data, K=K, initname='randexamples')
    return hmodel

def pprintProblemSpecStr(K=10, vocab_size=5000, nDocTotal=10, 
             nWordsPerDoc=200, LPkwargs=dict(), **kwargs):
    s = 'vocab_size=%d\nnDocTotal=%d\nnWordsPerDoc=%d\nK=%d\n' % (
        vocab_size, nDocTotal, nWordsPerDoc, K)
    for k,v in list(LPkwargs.items()):
        s += '%s=%f\n' % (k,v)
    return s

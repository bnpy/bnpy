'''
BarsK10V900.py

Toy Bars data, with K=10 topics and vocabulary size 900.
5 horizontal bars, and 5 vertical bars.

Generated via the standard LDA generative model
  see WordsData.CreateToyDataFromLDAModel for details.
'''
import numpy as np
from bnpy.data import WordsData
import Bars2D

SEED = 8675309
PRNG = np.random.RandomState(SEED)

# FIXED DATA GENERATION PARAMS
K = 10  # Number of topics
V = 1024  # Vocabulary Size
gamma = 0.85  # hyperparameter over doc-topic distribution

Defaults = dict()
Defaults['nDocTotal'] = 1000
Defaults['nWordsPerDoc'] = 300

# GLOBAL PROB DISTRIBUTION OVER TOPICS
B = 3.0
trueBeta = [B, 1, B, 1, B, 1, B, 1, B, 1.0]
trueBeta = np.asarray(trueBeta) / np.sum(trueBeta)
Defaults['topic_prior'] = gamma * trueBeta

# TOPIC by WORD distribution
Defaults['topics'] = Bars2D.Create2DBarsTopicWordParams2(
    V,
    K,
    r=0.1,
    PRNG=PRNG)


def get_data_info(**kwargs):
    if 'nDocTotal' in kwargs:
        nDocTotal = kwargs['nDocTotal']
    else:
        nDocTotal = Defaults['nDocTotal']
    return 'Toy Bars2 Data. Ktrue=%d. nDocTotal=%d.' % (K, nDocTotal)


def get_data(**kwargs):
    '''
        Args
        -------
        seed
        nDocTotal
        nWordsPerDoc
    '''
    Data = CreateToyDataFromLDAModel(seed=SEED, **kwargs)
    Data.summary = get_data_info(**kwargs)
    return Data


def CreateToyDataFromLDAModel(**kwargs):
    for key in Defaults:
        if key not in kwargs:
            kwargs[key] = Defaults[key]
    return WordsData.CreateToyDataFromLDAModel(**kwargs)

if __name__ == '__main__':
    import bnpy.viz.BarsViz
    from matplotlib import pylab
    WData = CreateToyDataFromLDAModel(seed=SEED)
    pylab.imshow(
        Defaults['topics'],
        aspect=V /
        float(K),
        interpolation='nearest')
    bnpy.viz.BarsViz.plotExampleBarsDocs(WData)
    pylab.show()

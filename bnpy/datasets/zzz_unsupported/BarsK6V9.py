'''
BarsK6V9.py

Toy Bars data, with K=6 topics and vocabulary size 9.
3 horizontal bars, and 3 vertical bars.

Generated via the standard LDA generative model
  see WordsData.CreateToyDataFromLDAModel for details.
'''
import numpy as np
from bnpy.data import WordsData
import Bars2D


def get_data_info():
    s = 'Toy Bars Data with %d true topics. Each doc uses 1-3 bars.' % (K)
    return s


def get_data(**kwargs):
    '''
        Args
        -------
        seed
        nDocTotal
        nWordsPerDoc
    '''
    Data = CreateToyDataFromLDAModel(seed=SEED, **kwargs)
    Data.name = 'BarsK6V9'
    Data.summary = get_data_info()
    return Data

SEED = 8675309
PRNG = np.random.RandomState(SEED)

# FIXED DATA GENERATION PARAMS
K = 6  # Number of topics
V = 9  # Vocabulary Size
gamma = 0.5  # hyperparameter over doc-topic distribution

Defaults = dict()
Defaults['nDocTotal'] = 200
Defaults['nWordsPerDoc'] = 25

# GLOBAL PROB DISTRIBUTION OVER TOPICS
trueBeta = np.ones(K)
trueBeta /= trueBeta.sum()
Defaults['topic_prior'] = gamma * trueBeta

# TOPIC by WORD distribution
Defaults['topics'] = Bars2D.Create2DBarsTopicWordParams(V, K, PRNG=PRNG)


def CreateToyDataFromLDAModel(**kwargs):
    for key in Defaults:
        if key not in kwargs:
            kwargs[key] = Defaults[key]
    return WordsData.CreateToyDataFromLDAModel(**kwargs)

if __name__ == '__main__':
    import bnpy.viz.BarsViz
    WData = CreateToyDataFromLDAModel(seed=SEED)
    bnpy.viz.BarsViz.plotExampleBarsDocs(WData)

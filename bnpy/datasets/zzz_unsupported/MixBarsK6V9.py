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

# FIXED DATA GENERATION PARAMS
K = 6  # Number of topics
V = 9  # Vocabulary Size
SEED = 8675309

Defaults = dict()
Defaults['seed'] = SEED
Defaults['nDocTotal'] = 200
Defaults['nWordsPerDoc'] = 25

# GLOBAL PROB DISTRIBUTION OVER TOPICS
trueBeta = np.ones(K)
trueBeta /= trueBeta.sum()
Defaults['beta'] = trueBeta

# TOPIC by WORD distribution
PRNG = np.random.RandomState(SEED)
Defaults['topics'] = Bars2D.Create2DBarsTopicWordParams(V, K, PRNG=PRNG)


def get_short_name():
    return 'MixBarsK6V9'


def get_data_info():
    s = 'Toy Bars Data with %d true topics. Each doc uses ONE topic.' % (K)
    return s


def get_data(**kwargs):
    ''' Create and return dataset.

        Keyword Args
        -------
        nDocTotal
        nWordsPerDoc
    '''
    updateKwArgsWithDefaults(kwargs)
    Data = WordsData.CreateToyDataFromMixModel(**kwargs)
    Data.name = get_short_name()
    Data.summary = get_data_info()
    return Data


def updateKwArgsWithDefaults(kwargs):
    for key in Defaults:
        if key not in kwargs:
            kwargs[key] = Defaults[key]

if __name__ == '__main__':
    import bnpy.viz.BarsViz
    WData = WordsData.CreateToyDataFromMixModel(**Defaults)
    bnpy.viz.BarsViz.plotExampleBarsDocs(WData)

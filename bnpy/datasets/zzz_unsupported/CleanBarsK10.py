'''
CleanBars

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
V = 900  # Vocabulary Size
gamma = 0.25  # hyperparameter over doc-topic distribution

Defaults = dict()
Defaults['nDocTotal'] = 1000
Defaults['nWordsPerDoc'] = 500

# GLOBAL PROB DISTRIBUTION OVER TOPICS
trueBeta = np.ones(K)
trueBeta /= trueBeta.sum()
Defaults['topic_prior'] = gamma * trueBeta

# TOPIC by WORD distribution
Defaults['topics'] = Bars2D.Create2DBarsTopicWordParams(
    V, K, PRNG=PRNG, fracMassOnTopic=0.999)


def get_data_info():
    s = 'Clean Bars Data with %d true topics. Each doc uses 1-3 bars.' % (K)
    return s

def get_data(seed=SEED, **kwargs):
    ''' Create toy dataset using bars topics.

    Keyword Args
    ------------
    seed : int
        Determines pseudo-random generator used to make the toy data.
    nDocTotal : int
        Number of total documents to create.
    nWordsPerDoc : int
        Number of total words to create in each document (all docs same length)
    '''
    Data = CreateToyDataFromLDAModel(seed=seed, **kwargs)
    Data.name = 'CleanBarsK10'
    Data.summary = get_data_info()
    return Data

def get_test_data(seed=6789, nDocTotal=100, **kwargs):
    ''' Create dataset of "heldout" docs, for testing purposes.

    Uses different random seed than get_data, but otherwise similar.
    '''
    Data = CreateToyDataFromLDAModel(seed=seed, nDocTotal=nDocTotal, **kwargs)
    Data.name = 'CleanBarsK10'
    Data.summary = get_data_info()
    return Data


def CreateToyDataFromLDAModel(**kwargs):
    for key in Defaults:
        if key not in kwargs:
            kwargs[key] = Defaults[key]
    return WordsData.CreateToyDataFromLDAModel(**kwargs)


def showTrueTopics(pylab=None, block=False):
    import bnpy.viz.BarsViz as BarsViz
    if pylab is not None:
        BarsViz.pylab = pylab
    BarsViz.showTopicsAsSquareImages(
        Defaults['topics'], vmin=0, vmax=np.percentile(Defaults['topics'], 95))
    BarsViz.pylab.show(block=block)

def showExampleDocs(pylab=None, block=False):
    import bnpy.viz.BarsViz as BarsViz
    WData = CreateToyDataFromLDAModel(seed=SEED)
    if pylab is not None:
        BarsViz.pylab = pylab
    BarsViz.plotExampleBarsDocs(WData)
    BarsViz.pylab.show(block=block)

def showDetailedPredictionsForDoc(Data, docID, vmax=4, block=False):
    docData = Data.select_subset_by_mask([docID])
    Info = IHT.calcPredLikForDoc(
        docData, Defaults['topics'],
        trueBeta, gamma)    
    titles = list()
    ImSet = np.zeros((4, Data.vocab_size), dtype=np.float64)
    titles.append("Complete doc")
    ImSet[0, docData.word_id] = docData.word_count
    titles.append("Training")
    ImSet[1, Info['tr_seen_wids']] = Info['tr_seen_wcts']
    titles.append("Heldout Labels")
    ImSet[2, Info['ho_seen_wids']] = Info['ho_seen_wcts']
    ImSet[2, Info['ho_unsn_wids']] = -1
    titles.append("Predictions")
    ho_all_wids = np.hstack([Info['ho_seen_wids'], Info['ho_unsn_wids']])
    scoreVec = Info['scoresOfHeldoutTypes']
    minScore = np.min(scoreVec)
    maxScore = np.percentile(scoreVec, 95)
    ImSet[3, ho_all_wids] = vmax / maxScore * (scoreVec - minScore)
    BarsViz.showTopicsAsSquareImages(ImSet, 
        vmin=-1, vmax=vmax, xlabels=titles)
    BarsViz.pylab.show(block=block)


if __name__ == '__main__':
    from bnpy.callbacks import InferHeldoutTopics as IHT

    Data = get_data(nDocTotal=10, nWordsPerDoc=500)
    RprecLabels = list()
    for rep in range(Data.nDoc):
        docData = Data.select_subset_by_mask([rep])
        Info = IHT.calcPredLikForDoc(
            docData, Defaults['topics'],
            trueBeta, gamma)
        print(" doc %d | Rprec %.3f AUC %.3f" % (
            rep, Info['R_precision'], Info['auc']))
        RprecLabels.append('%.3f' % (Info['R_precision']))
    import bnpy.viz.BarsViz as BarsViz
    BarsViz.plotExampleBarsDocs(Data,
        docIDsToPlot=np.arange(10),
        xlabels=RprecLabels,
        vmin=0,
        vmax=5,
        doShowNow=False)
    BarsViz.pylab.show(block=False)
    
    showDetailedPredictionsForDoc(Data, 0)
    showDetailedPredictionsForDoc(Data, 1)
    showDetailedPredictionsForDoc(Data, 2)
    showDetailedPredictionsForDoc(Data, 3, block=True)
    #showTrueTopics(block=True)
    #showExampleDocs(block=True)

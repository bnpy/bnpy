from builtins import *
import os
import numpy as np
import scipy.sparse
import time

from scipy.special import digamma

from .CPPLoader import LoadFuncFromCPPLib

curdir = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-1])
sparseLocalStepManyDocs_cpp = LoadFuncFromCPPLib(
    os.path.join(curdir, 'libsparsetopicsmanydocs.so'),
    os.path.join(curdir, 'TopicModelLocalStepManyDocsCPPX.cpp'),
    'sparseLocalStepManyDocs_ActiveOnly')

def sparseLocalStep_WordCountData(
        Data=None, LP=None,
        alphaEbeta=None, alphaEbetaRem=None,
        ElogphiT=None,
        DocTopicCount=None,
        spResp_data_OUT=None,
        spResp_colids_OUT=None,
        nCoordAscentItersLP=10,
        convThrLP=0.001,
        nnzPerRowLP=2,
        activeonlyLP=1,
        restartLP=0,
        restartNumTrialsLP=50,
        initDocTopicCountLP='setDocProbsToEGlobalProbs',
        reviseActiveFirstLP=-1,
        reviseActiveEveryLP=1,
        maxDiffVec=None,
        numIterVec=None,
        nRAcceptVec=None,
        nRTrialVec=None,
        verboseLP=0,
        **kwargs):
    ''' Perform local inference for topic model. Wrapper around C++ code.
    '''
    if LP is not None:
        ElogphiT = LP['ElogphiT']
    N = Data.nUniqueToken
    V, K = ElogphiT.shape
    assert K == alphaEbeta.size
    nnzPerRowLP = np.minimum(nnzPerRowLP, K)

    # Parse params for tracking convergence progress
    if maxDiffVec is None:
        maxDiffVec = np.zeros(Data.nDoc, dtype=np.float64)
        numIterVec = np.zeros(Data.nDoc, dtype=np.int32)
    if nRTrialVec is None:
        nRTrialVec = np.zeros(1, dtype=np.int32)
        nRAcceptVec = np.zeros(1, dtype=np.int32)
    assert maxDiffVec.dtype == np.float64
    assert numIterVec.dtype == np.int32

    # Handle starting from memoized doc-topic counts
    if initDocTopicCountLP == 'memo':
        if 'DocTopicCount' in LP:
            DocTopicCount = LP['DocTopicCount']
        else:
            initDocTopicCountLP = 'setDocProbsToEGlobalProbs'

    # Allow sparse restarts ONLY on first pass through dataset
    if restartLP > 1:
        if 'lapFrac' in kwargs and kwargs['lapFrac'] <= 1.0:
            restartLP = 1
        else:
            restartLP = 0

    # Use provided DocTopicCount array if its the right size
    # Otherwise, create a new one from scratch
    TopicCount_OUT = None
    if isinstance(DocTopicCount, np.ndarray):
        if DocTopicCount.shape == (Data.nDoc, K):
            TopicCount_OUT = DocTopicCount
    if TopicCount_OUT is None:
        TopicCount_OUT = np.zeros((Data.nDoc, K))
    assert TopicCount_OUT.shape == (Data.nDoc, K)
    if spResp_data_OUT is None:
        spResp_data_OUT = np.zeros(N * nnzPerRowLP)
        spResp_colids_OUT = np.zeros(N * nnzPerRowLP, dtype=np.int32)
    assert spResp_data_OUT.size == N * nnzPerRowLP
    assert spResp_colids_OUT.size == N * nnzPerRowLP

    if initDocTopicCountLP.startswith("setDocProbsToEGlobalProbs"):
        initProbsToEbeta = 1
    elif initDocTopicCountLP.startswith("fast"):
        initProbsToEbeta = 2
    else:
        initProbsToEbeta = 0
    if reviseActiveFirstLP < 0:
        reviseActiveFirstLP = nCoordAscentItersLP + 10

    sparseLocalStepManyDocs_cpp(
        alphaEbeta, ElogphiT,
        Data.word_count, Data.word_id, Data.doc_range,
        nnzPerRowLP, N, K, Data.nDoc, Data.vocab_size,
        nCoordAscentItersLP, convThrLP,
        initProbsToEbeta,
        TopicCount_OUT,
        spResp_data_OUT,
        spResp_colids_OUT,
        numIterVec,
        maxDiffVec,
        restartNumTrialsLP * restartLP,
        nRAcceptVec,
        nRTrialVec,
        reviseActiveFirstLP,
        reviseActiveEveryLP,
        verboseLP)

    # Package results up into dict
    if not isinstance(LP, dict):
        LP = dict()
    LP['nnzPerRow'] = nnzPerRowLP
    LP['DocTopicCount'] = TopicCount_OUT
    indptr = np.arange(
        0, (N+1) * nnzPerRowLP, nnzPerRowLP, dtype=np.int32)
    LP['spR'] = scipy.sparse.csr_matrix(
        (spResp_data_OUT, spResp_colids_OUT, indptr),
        shape=(N, K))
    # Fill in remainder of LP dict, with derived quantities
    from bnpy.allocmodel.topics.LocalStepManyDocs \
        import updateLPGivenDocTopicCount, writeLogMessageForManyDocs
    LP = updateLPGivenDocTopicCount(LP, LP['DocTopicCount'],
                                    alphaEbeta, alphaEbetaRem)
    LP['Info'] = dict()
    LP['Info']['iter'] = numIterVec
    LP['Info']['maxDiff'] = maxDiffVec

    if restartLP > 0:
        LP['Info']['nRestartsAccepted'] = nRAcceptVec[0]
        LP['Info']['nRestartsTried'] = nRTrialVec[0]
    writeLogMessageForManyDocs(Data, LP['Info'], LP,
        convThrLP=convThrLP, **kwargs)
    return LP

def doLocalStep_PythonLoopOverDocs(
        Data, model, **LPkwargs):
    LPkwargs['activeonlyLP'] = 1
    return model.calc_local_params(
        Data, **LPkwargs)

def doLocalStep_CPPLoopOverDocs(
        Data, model, **LPkwargs):
    LPkwargs['activeonlyLP'] = 2
    return model.calc_local_params(
        Data, **LPkwargs)
    '''
    return sparseLocalStep_WordCountData(
        Data,
        alphaEbeta=model.allocModel.alpha_E_beta(),
        alphaEbetaRem=model.allocModel.alpha_E_beta_rem(),
        ElogphiT=model.obsModel._E_logphiT('all'),
        **LPkwargs
        )
    '''
def compareLocalStep(Data, model, **LPkwargs):
    stime = time.time()
    LPold = doLocalStep_PythonLoopOverDocs(
        Data, model, **LPkwargs)
    pytime = time.time() - stime

    stime = time.time()
    LPnew = doLocalStep_CPPLoopOverDocs(
        Data, model, **LPkwargs)
    cpptime = time.time() - stime

    print("%8.3f sec | python" % (pytime))
    print("%8.3f sec | cpp" % (cpptime))
    print("Comparing Python vs C++ LocalStepManyDocs implementations...")
    print('    nDoc: %s' % (Data.nDoc))
    print('       K: %s' % (model.allocModel.K))
    for KwArgName in ['nCoordAscentItersLP', 'convThrLP', 'restartLP']:
        print('    %s: %s' % (KwArgName, LPkwargs[KwArgName]))
    for key in ['DocTopicCount', 'spR', 'iter']:
        compareLPValsAtKey(LPold, LPnew, key)
    return LPold, LPnew

def compareLPValsAtKey(LPold, LPnew, key):
    if key not in LPold:
        LPoldORIG = LPold
        LPnewORIG = LPnew
        LPold = LPold['Info']
        LPnew = LPnew['Info']
    assert key in LPold
    try:
        if key == 'spR':
            assert np.allclose(LPold[key].toarray(),
                               LPnew[key].toarray(), rtol=0, atol=.0001)
        else:
            assert np.allclose(LPold[key], LPnew[key], rtol=0, atol=.0001)
        print('  Good. Same value for LP[%s]' % (key))
    except AssertionError as e:
        print('  BAD!! Mismatch for LP[%s]' % (key))
        raise(e)


if __name__ == '__main__':
    import os
    os.environ['BNPYDATADIR'] = '/home/mhughes/git/x-topics/datasets/nips/'
    import bnpy

    import nips
    Data = nips.get_data()
    Data200 = Data.select_subset_by_mask(list(range(400)))
    Data200.name = 'nips400'
    model, Info = bnpy.run(Data200, 'HDPTopicModel', 'Mult', 'VB',
        nBatch=1, nLap=2, initname='randexamples', K=200,
        nCoordAscentItersLP=100, convThrLP=.01)

    '''
    model, Info = bnpy.run('BarsK10V900', 'HDPTopicModel', 'Mult', 'VB',
        nBatch=1, nDocTotal=50, nLap=2, initname='randexamples', K=200,
        nCoordAscentItersLP=100, convThrLP=.01)
    '''
    '''
    model, Info = bnpy.run('BarsK10V900', 'HDPTopicModel', 'Mult', 'VB',
        nBatch=1, nDocTotal=50, nLap=2, initname='truelabels',
        nCoordAscentItersLP=100, convThrLP=.01)
    '''
    # Find doc with significant usage by at least 2 topics
    Data = Info['Data']
    '''
    TrueDocTopicCount = np.zeros((Data.nDoc, Data.TrueParams['K']))
    for d in range(Data.nDoc):
        start_d = Data.doc_range[d]
        stop_d = Data.doc_range[d+1]
        wc_d = Data.word_count[start_d:stop_d]
        resp_d = Data.TrueParams['resp'][start_d:stop_d]
        TrueDocTopicCount[d] = np.dot(wc_d, resp_d)
    TrueDocTopicProb = TrueDocTopicCount / \
        TrueDocTopicCount.sum(axis=1)[:, np.newaxis]
    docsWithManyTopics = np.flatnonzero(TrueDocTopicProb.max(axis=1) < 0.6)
    assert len(docsWithManyTopics) > 0
    SingleDoc = Data.select_subset_by_mask(docsWithManyTopics[:1])
    '''

    '''
    LPkwargs = dict(
        nnzPerRowLP=4,
        initDocTopicCountLP='setDocProbsToEGlobalProbs',
        nCoordAscentItersLP=10,
        convThrLP=-1,
        restartLP=0,
        reviseActiveFirstLP=10,
        reviseActiveEveryLP=1)
    compareLocalStep(
        Data, model, **LPkwargs)

    LPkwargs = dict(
        nnzPerRowLP=4,
        initDocTopicCountLP='setDocProbsToEGlobalProbs',
        nCoordAscentItersLP=100,
        convThrLP=-1,
        restartLP=0,
        reviseActiveFirstLP=2,
        reviseActiveEveryLP=5)
    compareLocalStep(
        Data, model, **LPkwargs)

    LPkwargs = dict(
        nnzPerRowLP=4,
        initDocTopicCountLP='setDocProbsToEGlobalProbs',
        nCoordAscentItersLP=100,
        convThrLP=.1,
        restartLP=0,
        reviseActiveFirstLP=2,
        reviseActiveEveryLP=5)
    compareLocalStep(
        Data, model, **LPkwargs)


    LPkwargs = dict(
        nnzPerRowLP=6,
        initDocTopicCountLP='fastfirstiter_setDocProbsToEGlobalProbs',
        nCoordAscentItersLP=1,
        convThrLP=-1,
        restartLP=0,
        restartNumTrialsLP=50,
        reviseActiveFirstLP=2,
        reviseActiveEveryLP=5,
        verboseLP=0)
    _, LPcppfast = compareLocalStep(
        Data, model, **LPkwargs)

    LPkwargs = dict(
        nnzPerRowLP=6,
        initDocTopicCountLP='setDocProbsToEGlobalProbs',
        nCoordAscentItersLP=1, # will do TWO steps
        convThrLP=-1,
        restartLP=0,
        restartNumTrialsLP=50,
        reviseActiveFirstLP=2,
        reviseActiveEveryLP=5,
        verboseLP=0)
    _, LPcpp = compareLocalStep(
        Data, model, **LPkwargs)
    '''

    for reviseActiveEveryLP in [1, 10]:
      for nnzPerRow in [2, 4, 8, 16]:
        LPkwargs = dict(
            nnzPerRowLP=nnzPerRow,
            initDocTopicCountLP='fastfirstiter',
            nCoordAscentItersLP=100,
            convThrLP=0.01,
            restartLP=1,
            restartNumTrialsLP=50,
            reviseActiveFirstLP=5,
            reviseActiveEveryLP=1,
            activeonlyLP=2,
            verboseLP=0)
        print('>>> reviseActiveEveryLP=%d  nnzPerRow=%d' % (
            reviseActiveEveryLP, nnzPerRow))
        model.calc_local_params(Data, **LPkwargs)

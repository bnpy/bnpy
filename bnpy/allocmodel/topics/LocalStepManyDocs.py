from builtins import *
import numpy as np
import copy
import time

from scipy.special import digamma, gammaln
import scipy.sparse

from . import LocalStepLogger
from bnpy.util import NumericUtil
from .LocalStepSingleDoc import calcLocalParams_SingleDoc
from .LocalStepSingleDoc import calcLocalParams_SingleDoc_WithELBOTrace

from bnpy.util.SparseRespUtil \
    import fillInDocTopicCountFromSparseResp, sparsifyResp, sparsifyLogResp
from bnpy.util.lib.sparseResp.LibSparseResp \
    import calcSparseLocalParams_SingleDoc
from bnpy.util.lib.sparseResp.LibLocalStepManyDocs \
    import sparseLocalStep_WordCountData

def calcLocalParams(
        Data, LP,
        alphaEbeta=None,
        alphaEbetaRem=None,
        alpha=None,
        initDocTopicCountLP='scratch',
        cslice=(0, None),
        nnzPerRowLP=0,
        doSparseOnlyAtFinalLP=0,
        **kwargs):
    ''' Calculate all local parameters for provided dataset under a topic model

    Returns
    -------
    LP : dict
        Local parameter fields
        resp : 2D array, N x K
        DocTopicCount : 2D array, nDoc x K
        model-specific fields for doc-topic probabilities
    '''
    assert isinstance(cslice, tuple)
    if len(cslice) != 2:
        cslice = (0, None)
    elif cslice[0] is None:
        cslice = (0, None)
    nDoc = calcNumDocFromSlice(Data, cslice)

    if 'obsModelName' in LP:
        obsModelName = LP['obsModelName']
    elif hasattr(Data, 'word_count'):
        obsModelName = 'Mult'
    else:
        obsModelName = 'Gauss'
    # Unpack the problem size
    N, K = LP['E_log_soft_ev'].shape
    # Prepare the initial DocTopicCount matrix,
    # Useful for warm starts of the local step.
    initDocTopicCount = None
    if 'DocTopicCount' in LP:
        if LP['DocTopicCount'].shape == (nDoc, K):
            initDocTopicCount = LP['DocTopicCount'].copy()
    sumRespTilde = np.zeros(N)
    DocTopicCount = np.zeros((nDoc, K))
    DocTopicProb = np.zeros((nDoc, K))
    # Prepare the extra terms
    if alphaEbeta is None:
        assert alpha is not None
        alphaEbeta = alpha * np.ones(K)
    else:
        alphaEbeta = alphaEbeta[:K]
    # Prepare the likelihood matrix
    # Make sure it is C-contiguous, so that matrix ops are very fast
    Lik = np.asarray(LP['E_log_soft_ev'], order='C')
    if (nnzPerRowLP <= 0 or nnzPerRowLP >= K) or doSparseOnlyAtFinalLP:
        DO_DENSE = True
        # Dense Representation
        Lik -= Lik.max(axis=1)[:, np.newaxis]
        NumericUtil.inplaceExp(Lik)
    else:
        DO_DENSE = False
        nnzPerRowLP = np.minimum(nnzPerRowLP, K)
        spR_data = np.zeros(N * nnzPerRowLP, dtype=np.float64)
        spR_colids = np.zeros(N * nnzPerRowLP, dtype=np.int32)
    slice_start = Data.doc_range[cslice[0]]

    if not DO_DENSE and obsModelName.count('Mult'):
        if initDocTopicCountLP.count('fastfirstiter'):
            #tstart = time.time()
            init_spR = calcInitSparseResp(
                LP, alphaEbeta, nnzPerRowLP=nnzPerRowLP, **kwargs)
            #tstop = time.time()
            #telapsed = tstop - tstart

    AggInfo = dict()
    AggInfo['maxDiff'] = np.zeros(Data.nDoc)
    AggInfo['iter'] = np.zeros(Data.nDoc, dtype=np.int32)

    if 'restartLP' in kwargs and kwargs['restartLP']:
        AggInfo['nRestartsAccepted'] = np.zeros(1, dtype=np.int32)
        AggInfo['nRestartsTried'] = np.zeros(1, dtype=np.int32)
    else:
        AggInfo['nRestartsAccepted'] = None
        AggInfo['nRestartsTried'] = None

    for d in range(nDoc):
        start = Data.doc_range[cslice[0] + d]
        stop = Data.doc_range[cslice[0] + d + 1]
        if hasattr(Data, 'word_count') and obsModelName.count('Bern'):
            lstart = d * Data.vocab_size
            lstop = (d+1) * Data.vocab_size
        else:
            lstart = start - slice_start
            lstop = stop - slice_start
        if hasattr(Data, 'word_count') and not obsModelName.count('Bern'):
            wc_d = Data.word_count[start:stop].copy()
        else:
            wc_d = 1.0
        initDTC_d = None
        if initDocTopicCountLP == 'memo':
            if initDocTopicCount is not None:
                if DO_DENSE:
                    initDTC_d = initDocTopicCount[d]
                else:
                    DocTopicCount[d] = initDocTopicCount[d]
            else:
                initDocTopicCountLP = 'setDocProbsToEGlobalProbs'
        if not DO_DENSE and initDocTopicCountLP.count('fastfirstiter'):
            if obsModelName.count('Mult'):
                #tstart = time.time()
                DocTopicCount[d, :] = wc_d * init_spR[Data.word_id[start:stop]]
                #telapsed += time.time() - tstart
        if not DO_DENSE:
            m_start = nnzPerRowLP * start
            m_stop = nnzPerRowLP * stop

            # SPARSE RESP
            calcSparseLocalParams_SingleDoc(
                wc_d,
                Lik[lstart:lstop],
                alphaEbeta,
                topicCount_d_OUT=DocTopicCount[d],
                spResp_data_OUT=spR_data[m_start:m_stop],
                spResp_colids_OUT=spR_colids[m_start:m_stop],
                nnzPerRowLP=nnzPerRowLP,
                initDocTopicCountLP=initDocTopicCountLP,
                d=d,
                maxDiffVec=AggInfo['maxDiff'],
                numIterVec=AggInfo['iter'],
                nRAcceptVec=AggInfo['nRestartsAccepted'],
                nRTrialVec=AggInfo['nRestartsTried'],
                **kwargs)
        else:
            Lik_d = Lik[lstart:lstop].copy()  # Local copy
            (DocTopicCount[d], DocTopicProb[d],
                sumRespTilde[lstart:lstop], Info_d) \
                = calcLocalParams_SingleDoc(
                    wc_d, Lik_d, alphaEbeta, alphaEbetaRem,
                    DocTopicCount_d=initDTC_d,
                    initDocTopicCountLP=initDocTopicCountLP,
                    **kwargs)
            AggInfo = updateConvergenceInfoForDoc_d(d, Info_d, AggInfo, Data)
    #if initDocTopicCountLP.startswith('fast'):
    #    AggInfo['time_extra'] = telapsed
    LP['DocTopicCount'] = DocTopicCount
    if hasattr(Data, 'word_count'):
        if cslice is None or (cslice[0] == 0 and cslice[1] is None):
            assert np.allclose(np.sum(DocTopicCount), np.sum(Data.word_count))
    LP = updateLPGivenDocTopicCount(LP, DocTopicCount,
                                    alphaEbeta, alphaEbetaRem)
    if DO_DENSE:
        LP = updateLPWithResp(
            LP, Data, Lik, DocTopicProb, sumRespTilde, cslice,
            nnzPerRowLP=nnzPerRowLP,
            doSparseOnlyAtFinalLP=doSparseOnlyAtFinalLP)
    else:
        indptr = np.arange(
            0, (N+1) * nnzPerRowLP, nnzPerRowLP, dtype=np.int32)
        LP['spR'] = scipy.sparse.csr_matrix(
            (spR_data, spR_colids, indptr),
            shape=(N, K))
        LP['nnzPerRow'] = nnzPerRowLP

    LP['Info'] = AggInfo
    writeLogMessageForManyDocs(Data, AggInfo, LP, **kwargs)
    return LP

def calcInitSparseResp(LP, alphaEbeta, nnzPerRowLP=0, **kwargs):
    ''' Compute initial sparse responsibilities
    '''
    assert 'ElogphiT' in LP
    # Determine the top-L for each
    logS = LP['ElogphiT'].copy()
    logS += np.log(alphaEbeta)[np.newaxis,:]
    init_spR = sparsifyLogResp(logS, nnzPerRowLP)
    return init_spR

def updateLPGivenDocTopicCount(LP, DocTopicCount,
                               alphaEbeta, alphaEbetaRem=None):
    ''' Update local parameters given doc-topic counts for many docs.

    Returns for FiniteTopicModel (alphaEbetaRem is None)
    --------
    LP : dict of local params, with updated fields
        * theta : 2D array, nDoc x K
        * ElogPi : 2D array, nDoc x K

    Returns for HDPTopicModel (alphaEbetaRem is not None)
    --------
        * theta : 2D array, nDoc x K
        * ElogPi : 2D array, nDoc x K
        * thetaRem : scalar
        * ElogPiRem : scalar
    '''
    theta = DocTopicCount + alphaEbeta

    if alphaEbetaRem is None:
        # FiniteTopicModel
        digammaSumTheta = digamma(theta.sum(axis=1))
    else:
        # HDPTopicModel
        digammaSumTheta = digamma(theta.sum(axis=1) + alphaEbetaRem)
        LP['thetaRem'] = alphaEbetaRem
        LP['ElogPiRem'] = digamma(alphaEbetaRem) - digammaSumTheta
        LP['digammaSumTheta'] = digammaSumTheta  # Used for merges

    ElogPi = digamma(theta) - digammaSumTheta[:, np.newaxis]
    LP['theta'] = theta
    LP['ElogPi'] = ElogPi
    return LP


def updateLPWithResp(LP, Data, Lik, Prior, sumRespTilde,
        cslice=(0, None), doSparseOnlyAtFinalLP=0, nnzPerRowLP=0):
    ''' Compute assignment responsibilities given output of local step.

    Args
    ----
    LP : dict
        Has other fields like 'E_log_soft_ev'
    Data : DataObj
    Lik : 2D array, size N x K
        Will be overwritten and turned into resp.

    Returns
    -------
    LP : dict
        Add field 'resp' : N x K 2D array.
    '''
    # Create resp array directly from Lik array.
    # Do not make any copies, to save memory.
    LP['resp'] = Lik
    nDoc = calcNumDocFromSlice(Data, cslice)
    slice_start = Data.doc_range[cslice[0]]
    N = LP['resp'].shape[0]
    K = LP['resp'].shape[1]
    if N > Data.doc_range[-1]:
        assert N == nDoc * Data.vocab_size
        # Bernoulli naive case. Quite slow!
        for d in range(nDoc):
            rstart = d * Data.vocab_size
            rstop = (d+1) * Data.vocab_size
            LP['resp'][rstart:rstop] *= Prior[d]
    else:
        # Usual case. Quite fast!
        for d in range(nDoc):
            start = Data.doc_range[cslice[0] + d] - slice_start
            stop = Data.doc_range[cslice[0] + d + 1] - slice_start
            LP['resp'][start:stop] *= Prior[d]
    if doSparseOnlyAtFinalLP and (nnzPerRowLP > 0 and nnzPerRowLP < K):
        LP['spR'] = sparsifyResp(LP['resp'], nnzPerRow=nnzPerRowLP)
        LP['nnzPerRow'] = nnzPerRowLP
        assert np.allclose(LP['spR'].sum(axis=1), 1.0)
        del LP['resp']
        np.maximum(LP['spR'].data, 1e-300, out=LP['spR'].data)
        fillInDocTopicCountFromSparseResp(Data, LP)
    else:
        LP['resp'] /= sumRespTilde[:, np.newaxis]
        np.maximum(LP['resp'], 1e-300, out=LP['resp'])
    # Time consuming:
    # >>> assert np.allclose(LP['resp'].sum(axis=1), 1.0)
    return LP


def updateSingleDocLPWithResp(LP_d, Lik_d, Prior_d, sumR_d):
    resp_d = Lik_d.copy()
    resp_d *= Prior_d
    resp_d /= sumR_d[:, np.newaxis]
    np.maximum(resp_d, 1e-300, out=resp_d)
    LP_d['resp'] = resp_d
    return LP_d


def calcNumDocFromSlice(Data, cslice):
    if cslice[1] is None:
        nDoc = Data.nDoc
    else:
        nDoc = cslice[1] - cslice[0]
    return int(nDoc)


def writeLogMessageForManyDocs(Data, AI, LP,
                               sliceID=None,
                               **kwargs):
    """ Write log message summarizing convergence behavior across docs.

    Args
    ----
    Data : bnpy DataObj
    AI : dict of aggregated info for all documents.

    Post Condition
    --------------
    Message written to LocalStepLogger.
    """
    if 'lapFrac' not in kwargs:
        return
    if 'batchID' not in kwargs:
        return

    if isinstance(sliceID, int):
        sliceID = '%d' % (sliceID)
    else:
        sliceID = '0'

    perc = [0, 1, 10, 50, 90, 99, 100]
    siter = ' '.join(
        ['%d:%d' % (p, np.percentile(AI['iter'], p)) for p in perc])
    sdiff = ' '.join(
        ['%d:%.4f' % (p, np.percentile(AI['maxDiff'], p)) for p in perc])
    nConverged = np.sum(AI['maxDiff'] <= kwargs['convThrLP'])
    msg = 'lap %4.2f batch %d slice %s' % (
        kwargs['lapFrac'], kwargs['batchID'], sliceID)

    msg += ' nConverged %4d/%d' % (nConverged, AI['maxDiff'].size)
    worstDocID = np.argmax(AI['maxDiff'])
    msg += " worstDocID %4d \n" % (worstDocID)

    msg += ' iter prctiles %s\n' % (siter)
    msg += ' diff prctiles %s\n' % (sdiff)

    KactivePerDoc = np.sum(LP['DocTopicCount'] > .01, axis=1)
    sKactive = ' '.join(
        ['%d:%d' % (p, np.percentile(KactivePerDoc, p)) for p in perc])
    msg += ' Kact prctiles %s\n' % (sKactive)

    if 'nRestartsAccepted' in AI and AI['nRestartsAccepted'] is not None:
        msg += " nRestarts %4d/%4d\n" % (
            AI['nRestartsAccepted'], AI['nRestartsTried'])
    LocalStepLogger.log(msg)


def updateConvergenceInfoForDoc_d(d, Info_d, AggInfo, Data):
    """ Update convergence stats for specific doc into AggInfo.

    Returns
    -------
    AggInfo : dict, updated in place.
        * maxDiff : 1D array, nDoc
        * iter : 1D array, nDoc
    """
    if len(list(AggInfo.keys())) == 0:
        AggInfo['maxDiff'] = np.zeros(Data.nDoc)
        AggInfo['iter'] = np.zeros(Data.nDoc, dtype=np.int32)

    AggInfo['maxDiff'][d] = Info_d['maxDiff']
    AggInfo['iter'][d] = Info_d['iter']
    if 'ELBOtrace' in Info_d:
        AggInfo['ELBOtrace'] = Info_d['ELBOtrace']
    if 'nAccept' in Info_d:
        if 'nRestartsAccepted' not in AggInfo:
            AggInfo['nRestartsAccepted'] = 0
            AggInfo['nRestartsTried'] = 0
        AggInfo['nRestartsAccepted'] += Info_d['nAccept']
        AggInfo['nRestartsTried'] += Info_d['nTrial']
    return AggInfo

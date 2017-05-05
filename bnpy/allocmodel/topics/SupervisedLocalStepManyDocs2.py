import numpy as np
import copy
import time

from scipy.special import digamma, gammaln
import scipy.sparse

import LocalStepLogger
from bnpy.util import NumericUtil
from SupervisedLocalStepSingleDoc2 import calcLocalParams_SingleDoc
from SupervisedLocalStepSingleDoc2 import calcLocalParams_SingleDoc_WithELBOTrace

from bnpy.util.SparseRespUtil \
    import fillInDocTopicCountFromSparseResp, sparsifyResp, sparsifyLogResp
from bnpy.util.lib.sparseResp.LibSparseResp \
    import calcSparseLocalParams_SingleDoc


def calcLocalParams(
        Data, LP, eta=None, beta=None,
        alphaEbeta=None,
        alphaEbetaRem=None,
        alpha=None,
        delta=0.1,
        initDocTopicCountLP='scratch',
        cslice=(0, None),
        #nnzPerRowLP=0,
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
    else:
        LP['DocTopicCount'] = np.ones((nDoc, K)) / K

    # Initialize resp and theta for first pass to uniform
    if 'resp' in LP:
        resp = LP['resp']
        theta = LP['theta']
    else:
        resp = np.ones((int(Data.nUniqueToken), K)) * float(1.0/K)
        theta = np.ones((Data.nDoc, K)) + alpha
        #theta = np.ones((Data.nDoc, K)) * Data.nDoc/K

    resp_update = np.zeros(resp.shape)
    theta_update = np.zeros(theta.shape)

    DocTopicCount = np.ones((nDoc, K)) / K
    # DocTopicProb = np.zeros((nDoc, K))
    # Prepare the extra terms

    if alphaEbeta is None:
        assert alpha is not None
        alphaEbeta = alpha * np.ones(K)
    else:
        alphaEbeta = alphaEbeta[:K]

    # Prepare the likelihood matrix
    # Make sure it is C-contiguous, so that matrix ops are very fast
    Lik = np.asarray(LP['E_log_soft_ev'], order='C')
    DO_DENSE = True

    # Dense Representation
    Lik -= Lik.max(axis=1)[:, np.newaxis]
    NumericUtil.inplaceExp(Lik)

    slice_start = Data.doc_range[cslice[0]]

    AggInfo = dict()
    AggInfo['maxDiff'] = np.zeros(Data.nDoc)
    AggInfo['iter'] = np.zeros(Data.nDoc, dtype=np.int32)
    if 'restartLP' in kwargs and kwargs['restartLP']:
        AggInfo['nRestartsAccepted'] = np.zeros(1, dtype=np.int32)
        AggInfo['nRestartsTried'] = np.zeros(1, dtype=np.int32)
    else:
        AggInfo['nRestartsAccepted'] = None
        AggInfo['nRestartsTried'] = None

    for d in xrange(nDoc):
        start = Data.doc_range[cslice[0] + d]
        stop = Data.doc_range[cslice[0] + d + 1]

        wc_d = Data.word_count[start:stop]

        theta_d = theta[d]
        resp_d = resp[start:stop, :]
        Lik_d = Lik[start:stop, :]

        try:
            response_d = Data.response[d]
        except AttributeError:
            response_d = None
        resp_d_update, theta_d_update, DocTopicCount_d, Info_d = calcLocalParams_SingleDoc(
                        resp_d,theta_d,response_d,wc_d,
                        eta,Lik_d,
                        delta=delta,
        alpha=alpha,
        **kwargs)

        '''
        resp_d_update, theta_d_update, DocTopicCount_d, Info_d = calcLocalParams_SingleDoc_WithELBOTrace(
                resp_d,theta_d,response_d,wc_d,
                eta,Lik_d,
                delta=delta,alpha=alpha,**kwargs)
        '''

        resp_update[start:stop,:] = resp_d_update
        theta_update[d] = theta_d_update
        DocTopicCount[d,:] = DocTopicCount_d
        AggInfo = updateConvergenceInfoForDoc_d(d, Info_d, AggInfo, Data)

    LP['resp'] = resp_update
    LP['theta'] = theta_update
    digammaSumTheta = digamma(LP['theta'].sum(axis=1))
    LP['ElogPi'] = digamma(LP['theta']) - digammaSumTheta[:, np.newaxis]
    LP['Info'] = AggInfo
    LP['DocTopicCount'] = DocTopicCount
    writeLogMessageForManyDocs(Data, AggInfo, **kwargs)
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
    pass


def updateSingleDocLPWithResp(LP_d, Lik_d, Prior_d, sumR_d):
    pass


def calcNumDocFromSlice(Data, cslice):
    if cslice[1] is None:
        nDoc = Data.nDoc
    else:
        nDoc = cslice[1] - cslice[0]
    return int(nDoc)


def writeLogMessageForManyDocs(Data, AI,
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
    if len(AggInfo.keys()) == 0:
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

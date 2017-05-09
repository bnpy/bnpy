import numpy as np
import math
from numpy.linalg import inv

import itertools

from ..AllocModel import AllocModel
from bnpy.suffstats import SuffStatBag
from ...util import digamma, gammaln
from ...util import NumericUtil

from SupervisedLocalStepManyDocs2 import calcLocalParams


class SupervisedFiniteTopicModelWithPointEstimatedWeights(AllocModel):

    '''
    Bayesian topic model with a finite number of components K.

    Attributes
    -------
    inferType : string {'VB', 'moVB', 'soVB'}
        indicates which updates to perform for local/global steps
    K : int
        number of components
    alpha : float
        scalar pseudo-count
        used in Dirichlet prior on document -topic probabilities \pi_d.


    Attributes for VB
    ---------
    None. No global structure exists except scalar hypers alpha and delta.

    Variational Local Parameters
    --------
    resp :  2D array, N x K
        q(z_n) = Categorical( resp_{n1}, ... resp_{nK} )
    theta : 2D array, nDoc x K
        q(pi_d) = Dirichlet( \theta_{d1}, ... \theta_{dK} )

    References
    -------
    Supervised Latent Dirichlet Allocation, by McAulliffe and Blei
    introduces a supervised topic model with Dirichlet-Mult observations.
    '''

    def __init__(self, inferType, priorDict=None):
        if inferType == 'EM':
            raise ValueError('SupervisedFiniteTopicModel cannot do EM.')
        self.inferType = inferType
        self.K = 0
        self.eta = None
        if priorDict is None:
            self.set_prior()
        else:
            self.set_prior(**priorDict)

    def get_active_comp_probs(self):
        ''' Get vector of appearance probabilities for each active comp.

        Returns
        -------
        beta : 1D array, size K
            beta[k] gives probability of comp. k under this model.
        '''
        return np.ones(self.K) / float(self.K)

    def set_prior(self, alpha=1.0, delta=0.1,update_delta=0, **kwargs):
        self.alpha = float(alpha)
        self.delta = float(delta)
	self.update_delta = int(update_delta)

    def to_dict(self):
        return dict(eta=self.eta)

    def from_dict(self, Dict):
        self.inferType = Dict['inferType']
        self.K = Dict['K']
        self.eta = Dict['eta']

    def get_prior_dict(self):
        return dict(alpha=self.alpha,
                    K=self.K,
                    inferType=self.inferType,
                    eta=self.eta)

    def get_info_string(self):
        ''' Returns human-readable name of this object
        '''
        return 'Supervised Finite LDA model with K=%d comps. alpha=%.2f, delta=%.2f, update delta?=%d' \
            % (self.K, self.alpha, self.delta, self.update_delta)

    def calc_local_params(self, Data, LP, **kwargs):
        ''' Compute local parameters for each data item.

        Parameters
        -------
        Data : bnpy.data.DataObj subclass

        LP : dict
            Local parameters as key-value string/array pairs
            * E_log_soft_ev : 2D array, N x K
                E_log_soft_ev[n,k] = log p(data obs n | comp k)

        Returns
        -------
        LP : dict
            Local parameters, with updated fields
            * resp : 2D array, N x K
                Posterior responsibility each comp has for each item
                resp[n, k] = p(z[n] = k | x[n])
            * theta : 2D array, nDoc x K
                Defines approximate posterior on doc-topic weights.
                q(\pi_d) = Dirichlet(theta[d,0], ... theta[d, K-1])
        '''
	#print self.eta
        LP = calcLocalParams(
            Data, LP, eta=self.eta, alpha=self.alpha, delta=self.delta, **kwargs)
        assert 'resp' in LP
        assert 'theta' in LP
        assert 'DocTopicCount' in LP

        return LP

    def initLPFromResp(self, Data, LP):
        ''' Fill in remaining local parameters given token-topic resp.

        Args
        ----
        LP : dict with fields
            * resp : 2D array, size N x K

        Returns
        -------
        LP : dict with fields
            * DocTopicCount
            * theta
            * ElogPi
        '''

        resp = LP['resp']
        K = resp.shape[1]
        DocTopicCount = np.zeros((Data.nDoc, K))
        resp_idx_start = 0
        for d in xrange(Data.nDoc):
            start = Data.doc_range[d]
            stop = Data.doc_range[d + 1]
            DocTopicCount[d, :] = np.sum(resp[start:stop, :], axis=0)
        remMass = np.minimum(0.1, 1.0 / (K * K))
        newEbeta = (1 - remMass) / K
        theta = DocTopicCount + self.alpha * newEbeta
        digammaSumTheta = digamma(theta.sum(axis=1))
        ElogPi = digamma(theta) - digammaSumTheta[:, np.newaxis]

        LP['DocTopicCount'] = DocTopicCount
        LP['theta'] = theta
        LP['ElogPi'] = ElogPi

        return LP

    def get_global_suff_stats(self, Data, LP,
                              doPrecompEntropy=None,
                              cslice=(0, None), **kwargs):
        ''' Calculate sufficient statistics for global updates.

        Parameters
        -------
        Data : bnpy data object
        LP : local param dict with fields
            resp : Data.nObs x K array,
                where resp[n,k] = posterior resp of comp k
        doPrecompEntropy : boolean flag
            indicates whether to precompute ELBO terms in advance
            used for memoized learning algorithms (moVB)

        Returns
        -------
        SS : SuffStatBag with K components
            Relevant fields
            * nDoc : scalar float
                Counts total documents available in provided data.

            Also has optional ELBO field when precompELBO is True
            * Hvec : 1D array, size K
                Vector of entropy contributions from each comp.
                Hvec[k] = \sum_{n=1}^N H[q(z_n)], a function of 'resp'
        '''

        resp = LP['resp']
        nTokensTotal, K = resp.shape

        SS = SuffStatBag(K=K, D=Data.get_dim())
        if cslice[1] is None:
            SS.setField('nDoc', Data.nDoc, dims=None)
        else:
            SS.setField('nDoc', cslice[1] - cslice[0], dims=None)

        if doPrecompEntropy:
            Hvec = self.L_entropy(Data, LP, returnVector=1)
            Lalloc = self.L_alloc(Data, LP)
            Lslda = self.L_supervised(Data, LP)

            SS.setELBOTerm('Hvec', Hvec, dims='K')
            SS.setELBOTerm('L_alloc', Lalloc, dims=None)
            SS.setELBOTerm('L_sLDA', Lslda)

        SS.setField('vocab_size', Data.vocab_size, dims=None)
        SS.setField('response', Data.response, dims=(Data.nDoc,))
        SS.setField('doc_range', Data.doc_range, dims=Data.doc_range.shape)
        SS.setField('word_count', Data.word_count, dims=Data.word_count.shape)
        SS.setField('resp', resp, dims=(nTokensTotal, K))

        return SS

    def update_global_params(self, SS, rho=None, **kwargs):
        ''' Update global parameters to optimize the ELBO objective.
        '''
	if not self.update_delta:
		self.eta = update_global_params(
            SS.resp, SS.response, SS.doc_range, SS.word_count)
	else:
		self.eta, self.delta = update_global_params(
			SS.resp, SS.response, SS.doc_range, SS.word_count)
	self.K = SS.K

    def set_global_params(self, K=0, eta=None, **kwargs):
        """ Set global parameters to provided values.
        """
        print 'set global', kwargs
        self.K = K
        if eta is None:
            #self.eta = np.ones(K)
            self.eta = np.linspace(-1,1,K)
        else:
            self.eta = eta
        '''
        if hmodel is not None:
            self.setParamsFromHModel(hmodel)
        else:
            raise ValueError("Unrecognized set_global_params args")
        '''

    def init_global_params(self, Data, K=0, eta=None, **kwargs):
        """ Initialize global parameters to provided values.
        """
        self.K = K
        if eta is None:
            #self.eta = np.ones(K)
            self.eta = np.linspace(-1,1,K)
        else:
            self.eta = eta

    def setParamsFromHModel(self, hmodel):
        """ Set parameters exactly as in provided HModel object.

        Parameters
        ------
        hmodel : bnpy.HModel
            The model to copy parameters from.

        Post Condition
        ------
        Attributes rho/omega set exactly equal to hmodel's allocModel.
        """
        # self.ClearCache()
        self.K = hmodel.allocModel.K
        if hasattr(hmodel.allocModel, 'eta'):
            eta = hmodel.allocModel.eta.copy()
        else:
            raise AttributeError('Unrecognized hmodel')

    def calc_evidence(self, Data, SS, LP, **kwargs):
        """ Calculate ELBO objective function value for provided state.

        Returns
        -------
        L : float
            Represents sum of all terms in ELBO objective.
        """
        # print 'calc_evidence'
        if SS.hasELBOTerms():
            Lentropy = SS.getELBOTerm('Hvec').sum()
            Lalloc = SS.getELBOTerm('L_alloc')
            Lsupervised = SS.getELBOTerm('L_supervised')
        else:
            Lentropy = self.L_entropy(Data, LP, returnVector=0)
            Lalloc = self.L_alloc(Data, LP)
            Lsupervised = self.L_supervised(Data, LP)
        if SS.hasAmpFactor():
            Lentropy *= SS.ampF
            Lalloc *= SS.ampF
            Lsupervised *= SS.ampF
        return Lalloc + Lentropy + Lsupervised

    def L_entropy(self, Data, LP, returnVector=1):
        ''' Calculate assignment entropy term of the ELBO objective.

        Returns
        -------
        Hvec : 1D array, size K
            Hvec[k] = \sum_{n=1}^N H[q(z_n)]
        '''
        return L_entropy(Data, LP, returnVector=returnVector)

    def L_alloc(self, Data, LP):
        ''' Calculate allocation term of the ELBO objective.

        Returns
        -------
        L_alloc : scalar float
        '''
        return L_alloc(Data=Data, LP=LP, alpha=self.alpha)

    def L_supervised(self, Data, LP):
        ''' Calculate supervised term of the ELBO objective.

        Returns
        -------
        L_supervised : scalar float
        '''
        return L_supervised(Data=Data, LP=LP, eta=self.eta, delta=self.delta)

    def getSerializableParamsForLocalStep(self):
        """ Get compact dict of params for parallel local step.

        Returns
        -------
        Info : dict
        """
        return dict(inferType=self.inferType,
                    K=self.K,
                    alpha=self.alpha,
                    delta=self.delta,
                    eta=self.eta)

    def fillSharedMemDictForLocalStep(self, ShMem=None):
        """ Get dict of shared mem arrays needed for parallel local step.

        Returns
        -------
        ShMem : dict of RawArray objects
        """
        # No shared memory required here.
        if isinstance(ShMem, dict):
            return ShMem
        else:
            return dict()

    def getLocalAndSummaryFunctionHandles(self):
        """ Get function handles for local step and summary step

        Useful for parallelized algorithms.

        Returns
        -------
        calcLocalParams : f handle
        calcSummaryStats : f handle
        """
        return calcLocalParams, calcSummaryStats


def L_alloc(Data=None, LP=None, nDoc=0, alpha=1.0, **kwargs):
    ''' Calculate allocation term of the ELBO objective.

    E[ log p(pi) + log p(z) - log q(pi)  ]

    Returns
    -------
    L_alloc : scalar float
    '''

    if Data is not None:
        nDoc = Data.nDoc
    if LP is None:
        LP = dict(**kwargs)

    K = LP['DocTopicCount'].shape[1]
    cDiff = nDoc * c_Func(alpha, K) - c_Func(LP['theta'])
    slackVec = LP['DocTopicCount'] + alpha - LP['theta']
    slackVec *= LP['ElogPi']

    return cDiff + np.sum(slackVec)


def L_entropy(Data=None, LP=None, resp=None, returnVector=0):
    """ Calculate entropy of soft assignments term in ELBO objective.

    Returns
    -------
    L_entropy : scalar float
    """

    if LP is not None:
        resp = LP['resp']

    if hasattr(Data, 'word_count') and resp.shape[0] == Data.nUniqueToken:
        Hvec = -1 * NumericUtil.calcRlogRdotv(resp, Data.word_count)
    else:
        Hvec = -1 * NumericUtil.calcRlogR(resp)

    assert Hvec.min() >= -1e-6
    if returnVector:
        return Hvec

    return Hvec.sum()


def L_supervised(Data=None, LP=None, eta=None, delta=None):
    """Calculate slda term of the ELBO objective.

    E[p(y)]

    Returns
    -------
    L_supervised : scalar float
    """
    if LP is not None:
        resp = LP['resp']

    response = Data.response

    L_supervised = 0

    nDoc = Data.nDoc

    for d in xrange(nDoc):
        start = Data.doc_range[d]
        stop = Data.doc_range[d+1]

        wc_d = Data.word_count[start:stop]
        N_d = sum(wc_d)
        resp_d = np.asarray(resp[start:stop, :])

        response_d = response[d]
        nTokens_d, K = resp_d.shape

        weighted_resp_d = wc_d[:, None] * resp_d
        EZ_d = np.sum(weighted_resp_d, axis=0) / float(N_d)

        EZTZ_d = calc_EZTZ_one_doc(wc_d, resp_d)

        sterm = np.dot(eta, EZTZ_d)
        sterm = np.dot(sterm, eta)

        L_supervised_d = (-0.5) * np.log(2.0 * math.pi * delta)
        L_supervised_d -= np.square(response_d) / (2.0 * delta)
        L_supervised_d += (response_d / delta) * np.inner(eta, EZ_d)
        L_supervised_d -= sterm / (2.0 * delta)

        L_supervised += L_supervised_d

    return L_supervised


def c_Func(avec, K=0):
    ''' Evaluate cumulant function of the Dirichlet distribution

    Returns
    -------
    c : scalar real
    '''
    if isinstance(avec, float) or avec.ndim == 0:
        assert K > 0
        avec = avec * np.ones(K)
        return gammaln(np.sum(avec)) - np.sum(gammaln(avec))
    elif avec.ndim == 1:
        return gammaln(np.sum(avec)) - np.sum(gammaln(avec))
    else:
        return np.sum(gammaln(np.sum(avec, axis=1))) - np.sum(gammaln(avec))


def calcSummaryStats(Dslice, LP=None, alpha=None,
                     doPrecompEntropy=0,
                     **kwargs):
    """ Calculate summary from local parameters for given data slice.

    Parameters
    -------
    Data : bnpy data object
    LP : local param dict with fields
        resp : Data.nObs x K array,
            where resp[n,k] = posterior resp of comp k
        doPrecompEntropy : boolean flag
            indicates whether to precompute ELBO terms in advance
            used for memoized learning algorithms (moVB)

    Returns
    -------
    SS : SuffStatBag with K components
        * nDoc : scalar float
            Counts total documents available in provided data.

        Also has optional ELBO field when precompELBO is True
        * Hvec : 1D array, size K
            Vector of entropy contributions from each comp.
            Hvec[k] = \sum_{n=1}^N H[q(z_n)], a function of 'resp'
    """
    resp = LP['resp']
    _, K = resp.shape

    SS = SuffStatBag(K=K, D=Dslice.dim)
    SS.setField('nDoc', Dslice.nDoc, dims=None)
    if doPrecompEntropy:
        Hvec = L_entropy(Dslice, LP, returnVector=1)
        Lalloc = L_alloc(Dslice, LP, alpha=alpha)
        Lslda = L_sLDA(Dslice, LP)
        SS.setELBOTerm('Hvec', Hvec, dims='K')
        SS.setELBOTerm('L_alloc', Lalloc, dims=None)
        SS.setELBOTerm('Lslda', Lslda)
    return SS


def update_global_params(resp, response, doc_range, word_count):
    _, K = resp.shape
    if response is None:
        return np.zeros(K)

    nDoc = response.shape[0]

    # Update eta (response weights per topic)
    EX = np.zeros((nDoc,K)) # E[X], X[d] = \bar{Z}_d, D X K
    EXTX = np.zeros((K,K)) #E[X^T X], KxK

    for d in xrange(nDoc):
        start = int(doc_range[d])
        stop = int(doc_range[d+1] )
        wc_d = word_count[start:stop]

        N_d = int(sum(wc_d))

        resp_d = resp[start:stop,:]
        nTokens_d,_ = resp_d.shape

        weighted_resp_d = wc_d[:,None] * resp_d
        EX[d,:] = (1 / float(N_d)) * np.sum(weighted_resp_d,axis=0)


        EXTX += calc_EZTZ_one_doc(wc_d,resp_d)


    EXTXinv = inv(EXTX)

    # Update eta
    eta_update = np.dot(EXTXinv,EX.transpose())
    eta_update = np.dot(eta_update,response)

    return eta_update

def update_global_params2(resp, response, doc_range, word_count):
    _, K = resp.shape
    if response is None:
        return np.zeros(K)

    nDoc = response.shape[0]

    # Update eta (response weights per topic)
    EX = np.zeros((nDoc,K)) # E[X], X[d] = \bar{Z}_d, D X K
    EXTX = np.zeros((K,K)) #E[X^T X], KxK

    for d in xrange(nDoc):
        start = int(doc_range[d])
        stop = int(doc_range[d+1] )
        wc_d = word_count[start:stop]

        N_d = int(sum(wc_d))

        resp_d = resp[start:stop,:]
        nTokens_d,_ = resp_d.shape

        weighted_resp_d = wc_d[:,None] * resp_d
        EX[d,:] = (1 / float(N_d)) * np.sum(weighted_resp_d,axis=0)


        EXTX += calc_EZTZ_one_doc(wc_d,resp_d)


    EXTXinv = inv(EXTX)

    # Update eta
    eta_update = np.dot(EXTXinv,EX.transpose())
    eta_update = np.dot(eta_update,response)

    # Update delta
    yEX = np.dot(response,EX)
    delta_update = np.dot(yEX,eta_update)
    delta_update = np.inner(response,response) - delta_update
    delta_update = delta_update / np.float(nDoc)

    return eta_update, delta_update



def calc_EZTZ_one_doc(wc_d,resp_d):


    nTokens_d, K = resp_d.shape
    weighted_resp_d = wc_d[:,None] * resp_d #Doc
    N_d = np.sum(wc_d)

    A = np.ones((nTokens_d,nTokens_d)) - np.eye(nTokens_d)

    EZTZ = np.inner(weighted_resp_d.transpose(),A)
    EZTZ = np.inner(EZTZ,weighted_resp_d.transpose())

    EZTZ = EZTZ + np.diag(np.sum(wc_d[:,None] * weighted_resp_d,axis=0))


    '''

    SLOW WAY:

    EZTZ = np.zeros((K,K))

    for t,s in itertools.combinations(xrange(nTokens_d), 2):
        tmp_KK = np.outer(weighted_resp_d[t], weighted_resp_d[s])
        EZTZ += tmp_KK
        EZTZ += tmp_KK.T

    for t in xrange(nTokens_d):
        EZTZ += (wc_d[t] * wc_d[t]) * np.diag(resp_d[t])

    '''

    EZTZ = (1.0/np.square(N_d)) * EZTZ


    return EZTZ

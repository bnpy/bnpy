import numpy as np
import scipy.linalg
from scipy.special import gammaln, digamma
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from bnpy.suffstats import ParamBag, SuffStatBag
from bnpy.util import LOGTWO, LOGPI, LOGTWOPI, EPS
from bnpy.util import dotATA, dotATB, dotABT
from bnpy.util import as1D, as2D, as3D, toCArray, np2flatstr
from bnpy.util import numpyToSharedMemArray, fillSharedMemArray
from bnpy.util.SparseRespStatsUtil import calcSpRXXT
from AbstractObsModel import AbstractObsModel

import matplotlib.pyplot as plt

class SMLogisticRegressYFromFixedTopicModelDiag(AbstractObsModel):

    ''' Model for producing 1D observations from fixed covariates

    Attributes for Prior
    --------------------
    w_E : 1D array, size E
        mean of the regression weights
    P_EE : 2D array, size E x E
        precision matrix for regression weights
    pnu : positive float
        effective sample size of prior on regression precision
    ptau : positive float
        effective scale parameter of prior on regression precision

    Attributes for Point Estimation
    -------------------------------
    TODO

    Attributes for Approximate Posterior
    ------------------------------------
    w_E : 1D array, size E
    P_EE : 2D array, size E x E
    pnu : positive float
    ptau : positive float
    '''

    def __init__(self, inferType='VB', D=0, Data=None, **PriorArgs):
        ''' Initialize bare obsmodel with valid prior hyperparameters.

        Returns
        -------
        obsmodel : bare observation model
            Resulting object lacks either EstParams or Post attributes.
            which must be created separately (see init_global_params).
        '''
        self.D = 1
        self.E = 2
        self.K = 0
        self.inferType = inferType
        self.Prior = createParamBagForPrior(Data, **PriorArgs)
        self.Cache = dict()

    def get_name(self):
        return 'LogisticRegressYFromFixedTopics'

    def get_info_string(self):
        return 'Logistic regression model for binary y from topic proportions'

    def get_info_string_prior(self):
        return getStringSummaryOfPrior(self.Prior)


    def setPostFactors(self, obsModel=None, SS=None, LP=None, Data=None,
                       **param_kwargs):
        ''' Set attribute Post to provided values.
        '''
        self.ClearCache()
        if obsModel is not None:
            if hasattr(obsModel, 'Post'):
                self.Post = obsModel.Post.copy()
                self.K = self.Post.K
            else:
                self.setPostFromEstParams(obsModel.EstParams)
            return

        if LP is not None and Data is not None:
            SS = self.calcSummaryStats(Data, None, LP)
        if SS is not None:
            self.updatePost(SS)
        else:
            self.Post = packParamBagForPost(**param_kwargs)
        self.K = self.Post.K

    def calcSummaryStats(self, Data, SS, LP, **kwargs):
        ''' Calculate summary statistics for given dataset and local parameters

        Returns
        --------
        SS : SuffStatBag object, with K components.
        '''
        Post = None if not hasattr(self, 'Post') else self.Post
        return calcSummaryStats(Data, SS, LP, Prior=self.Prior, Post=Post **kwargs)

    def getDatasetScale(self, SS):
        ''' Get number of observed scalars in dataset from suff stats.

        Used for normalizing the ELBO so it has reasonable range.

        Returns
        ---------
        s : scalar positive integer
        '''
        return SS.N.sum()

    def calcLogSoftEvMatrix_FromPost(self, Data, **kwargs):
        ''' Compute expected log soft evidence of each item under each cluster

        Returns
        -------
        E_log_soft_ev_NK : 2D array, size N x K
        '''
        return calcLogSoftEvMatrix_FromPost(
            Data,
            m=self.Post.w_m,
            Sinv=self.Post.Sinv,
            S=self.Post.S, 
            **kwargs)

    def updatePost(self, SS):
        ''' Update attribute Post for all comps given suff stats.

        Optimizes the variational objective for approximating the posterior

        Post Condition
        --------------
        Attributes K and Post updated in-place.
        '''
        self.ClearCache()
        if not hasattr(self, 'Post') or self.Post.K != SS.K:
            self.Post = ParamBag(K=SS.K, D=SS.D, E=SS.D+1)
        self.Post = calcPostParamsFromSS(
            SS=SS, Prior=self.Prior, returnParamBag=True)
        self.K = SS.K


    def calcELBO_Memoized(self, SS, returnVec=0, afterMStep=False, **kwargs):
        """ Calculate obsModel's objective using suff stats SS and Post.

        Args
        -------
        SS : bnpy SuffStatBag
        afterMStep : boolean flag
            if 1, elbo calculated assuming M-step just completed

        Returns
        -------
        elbo_K : scalar float
            Equal to E[ log p(x) + log p(phi) - log q(phi)]
        """
        return calcELBOFromSSAndPost(
            SS=SS,
            Post=self.Post,
            Prior=self.Prior,
            returnVec=returnVec,
            afterMStep=afterMStep)


    def calcHardMergeGap(self, SS, kA, kB):
        ''' Calculate change in ELBO after a hard merge applied to this model

        Returns
        ---------
        gap : scalar real, indicates change in ELBO after merge of kA, kB
        '''
        gap, _, _ = calcHardMergeGapForPair(
            SS=SS, Post=self.Post, Prior=self.Prior, kA=kA, kB=kB)
        return gap

    def calcHardMergeGap_SpecificPairs(self, SS, PairList):
        ''' Calc change in ELBO for specific list of candidate hard merge pairs

        Returns
        ---------
        Gaps : 1D array, size L
              Gap[j] : scalar change in ELBO after merge of pair in PairList[j]
        '''
        Gaps = np.zeros(len(PairList))
        cPrior = None
        cPost_K = [None for k in range(SS.K)]
        for ii, (kA, kB) in enumerate(PairList):
            Gaps[ii], cPost_K, cPrior = calcHardMergeGapForPair(
                SS=SS, Post=self.Post, Prior=self.Prior, kA=kA, kB=kB,
                cPrior=cPrior, cPost_K=cPost_K)
        return Gaps

''' Functions for computing the updates and ELBO terms for the variational 
    logistic regression model.
'''


def lam(eta):
    return np.tanh(eta / 2.0) / (4.0 * eta)

def calc_sinv_ss(eta, X):
    return np.dot((lam(eta)) * X.T, X)

def sinv_update(siginv, sinv_ss):
    return siginv + 2.0 * np.diag(sinv_ss)

def calc_m_ss(X, y):
    return np.dot(((y - 0.5)), X)

def m_update(m_ss, sinv_ss, mu, siginv):
    si = np.eye(m_ss.shape[0]) * siginv
    return np.linalg.solve(si + 2 * sinv_ss, (siginv * mu) + m_ss)

def eta_update(m, S, X):
    m, S = np.asarray(m).reshape((-1,)), np.asarray(S).reshape((-1,))
    if m.shape[0] < X.shape[1]:
        m = np.ones(X.shape[1]) * m[0]
        S = np.ones(X.shape[1]) * S[0]

    eta2 = np.dot(X ** 2, S) + (np.dot(X, m) ** 2)
    return np.sqrt(eta2)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def log_g(x):
    return -np.log(1.0 + np.exp(-x))

def calc_eta_ss(eta):
    return np.sum(log_g(eta) -  0.5 * eta + lam(eta) * (eta ** 2))

def exp_log_lik_bound_vec(X, y, m, S, eta):
    Xm = np.dot(X, m)
    ellik = log_g(eta) -  0.5 * eta + lam(eta) * (eta ** 2)
    ellik = ellik + (y - 0.5) * Xm
    ellik = ellik - lam(eta) * (Xm ** 2 + np.dot(X ** 2, S))
    return ellik

def exp_log_lik_bound(m_ss, sinv_ss, eta_ss, m, S):
    bound = eta_ss
    bound = bound + np.dot(m_ss, m) 
    bound = bound - np.sum(sinv_ss * np.outer(m, m)) - np.dot(np.diag(sinv_ss), S)
    return bound

def exp_log_prior(m, S, siginv, sig):
    if not hasattr(sig, 'shape'):
        siginv = np.ones(m.shape) * siginv
        sig = np.ones(m.shape) * sig

    elp = -0.5 * (np.sum(np.log(2 * np.pi * sig)))
    elp = elp - 0.5 * np.sum((m ** 2) * siginv)
    elp = elp - 0.5 * np.sum(siginv * S)

    return elp

def exp_log_entropy(S):
    return 0.5 * np.sum(np.log((2 * np.pi * np.e * S)))
    
#TODO: thid is wrong for nonzero mean w pror
def elbo(m_ss, sinv_ss, eta_ss, m, S, siginv, sig):
    elbo = exp_log_lik_bound(m_ss, sinv_ss, eta_ss, m, S)
    elbo = elbo + exp_log_prior(m, S, siginv, sig)
    elbo = elbo + exp_log_entropy(S)

    return elbo

''' Other module-level functions
'''

def calcLocalParams(Dslice, Post=None,
        Prior=None,
        m=None,
        Sinv=None,
        S=None,
        E_log_soft_ev_NK=None,
        **kwargs):

    LP = dict()
    LP['supervised'] = True
    if Post is None or not hasattr(Post, 'w_m'):
        LP['w_m'] = Prior.mu
        LP['w_var'] = Prior.sig
    else:
        LP['w_m'] = Post.w_m
        LP['w_var'] = Post.S
    return LP


def calcLogSoftEvMatrix_FromPost(
        Dslice,
        Post=None,
        m=None,
        Sinv=None,
        S=None,
        E_log_soft_ev_NK=None,
        **kwargs):
    ''' Calculate expected log soft ev matrix under approximate posterior

    Returns
    -------
    E_log_soft_ev_NK : 2D array, size N x K
    '''        
    assert False
    return 0


def calcSummaryStats(Data, SS, LP, Prior=None, Post=None, **kwargs):
    ''' Calculate summary statistics for given dataset and local parameters

    Returns
    --------
    SS : SuffStatBag object, with K components.
    '''
    #Setup the data from the (normalized) observed token assignments
    Ndoc = LP['DocTopicCount'].shape[0]
    X = LP['DocTopicCount']
    X = X / np.sum(X, axis=1).reshape((-1, 1))

    Y = Data.Y

    if SS is None:
        SS = SuffStatBag(K=K, D=Data.dim)

    
    if Post is None or not hasattr(Post, 'w_m'):
        eta = eta_update(Prior.mu, Prior.sig, X)
        eta_ss = calc_eta_ss(eta)
    else:
        eta = eta_update(Post.w_m, Post.S, X)
        eta_ss = calc_eta_ss(eta)

    m_ss = calc_m_ss(X, Y)
    sinv_ss = calc_sinv_ss(eta, X)

    for i in range(20):
        Sinv_post = sinv_update(Prior.siginv, sinv_ss)
        m_post = m_update(m_ss, sinv_ss, Prior.mu, Prior.siginv)
        
        eta = eta_update(m_post, 1.0 / Sinv_post, X)
        eta_ss = calc_eta_ss(eta)

        if Data.nDoc != Data.nDocTotal:
            break

        m_ss = calc_m_ss(X, Y)
        sinv_ss = calc_sinv_ss(eta, X)
        

    SS.setField('m_ss', m_ss, dims=('K'))
    SS.setField('sinv_ss', sinv_ss, dims=('K', 'K'))
    SS.setField('eta_ss', eta_ss, dims=None)
    # Expected count for each k
    # Usually computed by allocmodel. But just in case...
    if not hasattr(SS, 'N'):
        if 'resp' in LP:
            SS.setField('N', LP['resp'].sum(axis=0), dims='K')
        else:
            SS.setField('N', as1D(toCArray(LP['spR'].sum(axis=0))), dims='K')

    return SS


def calcPostParamsFromSS(
        SS=None, m_ss=None, sinv_ss=None, eta_ss=None, N_K=None,
        Prior=None,
        Post=None,
        returnParamBag=True,
        **kwargs):
    ''' Calc updated posterior parameters for all clusters from suff stats

    Returns
    --------
    pnu_K : 1D array, size K
    ptau_K : 1D array, size K
    w_KE : 2D array, size K x E
    P_KEE : 3D array, size K x E x E
    '''
    K = SS.K
    m_ss = SS.m_ss
    sinv_ss = SS.sinv_ss

    Sinv_post = sinv_update(Prior.siginv, sinv_ss)
    m_post = m_update(m_ss, sinv_ss, Prior.mu, Prior.siginv)

    if not returnParamBag:
        return m_post, Sinv_post
    return packParamBagForPost(
        m=m_post,
        Sinv=Sinv_post,
        Post=Post)

def calcPostParamsFromSSForComp(
        SS=None, kA=0, kB=None,
        Prior=None,
        **kwargs):
    ''' Calc posterior parameters for specific cluster from SS

    Returns
    --------
    pnu_K : float
    ptau_K : float
    w_KE : 1D array, size E
    P_KEE : 2D array, size E x E
    '''
    assert False

def calcELBOFromSSAndPost(
        SS, Post=None, Prior=None,
        returnVec=0, afterMStep=False, **kwargs):
    """ Calculate obsModel objective function using suff stats SS and Post.

    Args
    -------
    SS : bnpy SuffStatBag
    Post : bnpy ParamBag
    afterMStep : boolean flag
        if 1, elbo calculated assuming M-step just completed

    Returns
    -------
    elbo_K : scalar float
        Equal to E[ log p(x) + log p(phi) - log q(phi)]
    """
    elbo_K = 1 * elbo(SS.m_ss, SS.sinv_ss, SS.eta_ss, Post.w_m, Post.S, Prior.siginv, Prior.sig)

    if returnVec:
        return elbo_K * np.ones((SS.K,)) / SS.K
    return elbo_K.sum()


def packParamBagForPost(
        m=None,
        Sinv=None,
        Post=None,
        **kwargs):
    ''' Parse provided array args and pack into parameter bag

    Returns
    -------
    Post : ParamBag, with K clusters
    '''
    K = m.shape[0]
    S = 1.0 / Sinv

    m = as1D(m)
    Sinv = as1D(Sinv)
    S = as1D(S)
    
    if Post is None:
        Post = ParamBag(K=K)
    assert Post.K == K

    Post.setField('w_m', m, dims=('K',))
    Post.setField('Sinv', Sinv, dims=('K',))
    Post.setField('S', S, dims=('K',))
    return Post

def getStringSummaryOfPrior(Prior):
    ''' Create string summarizing prior information

    Returns
    -------
    s : str
    '''
    msg = 'Diagonal Gaussian prior on regression weights\n'
    return msg

def createParamBagForPrior(
        Data=None, D=0,
        w_E=0,
        P_EE=None, P_diag_val=1.0,
        Prior=None,
        **kwargs):
    ''' Initialize Prior ParamBag attribute.

    Returns
    -------
    Prior : ParamBag
        with dimension attributes K, D, E
        with parameter attributes pnu, ptau, w_E, P_EE
    '''
    if Data is None:
        D = int(D)
    else:
        D = int(Data.dim)

    # Initialize precision matrix of the weight vector
    P_EE = P_diag_val
    P_EE_inv = 1.0 / P_EE

    if Prior is None:
        Prior = ParamBag(K=0, D=D)

    Prior.setField('mu', w_E, dims=None)
    Prior.setField('sig', P_EE, dims=None)
    Prior.setField('siginv', P_EE_inv, dims=None)

    return Prior

def calcHardMergeGapForPair(
        SS=None, Prior=None, Post=None, kA=0, kB=1, 
        cPost_K=None,
        cPrior=None,
        ):
    ''' Compute difference in ELBO objective after merging two clusters

    Uses caching if desired

    Returns
    -------
    Ldiff : scalar
        difference in ELBO from merging cluster indices kA and kB
    '''
    Ev = elbo(SS.m_ss, SS.sinv_ss, SS.eta_ss, Post.w_m, Post.S, Prior.siginv, Prior.sig)

    m_ss = np.delete(SS.m_ss, kB)
    m_ss[kA] += SS.m_SS[kB]
    sinv_ss = np.delete(SS.sinv_ss, kB)
    sinv_ss[kA] += SS.sinv_ss[kB]

    Sinv_post = sinv_update(Prior.siginv, sinv_ss)
    s_post = 1.0 / Sinv_post
    m_post = m_update(Sinv_post, m_ss, Prior.mu, Prior.siginv)

    mEv = elbo(m_ss, sinv_ss, eta_ss, m_post, s_post, Prior.siginv, Prior.sig)
    Gap = mEv - Ev

    return Gap, None, None

